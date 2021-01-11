import os
import six
import sys
import json
import math
import argparse
import matplotlib
import numpy as np
import collections
import chainer.optimizers as O
import chainer.links as L
import chainer.initializers as I
import chainer.functions as F
import chainer
from chainer.optimizer_hooks import GradientClipping
from chainer.training import extensions, triggers, Trainer
from chainer.backends import cuda
from chainer import reporter, training
from model_reader import ModelReader
from data_iterator import (
    SentenceIterator,
    SentenceDynamicIterator,
    SentimentSentenceIterator,
    SentimentSentenceDynamicIterator,
)
from data_loader import DataLoader
from prepare_opinion_contextualization_data import read_and_trim_vocab, get_aspect_opinions
from random_seed import set_random_seed
from defs import *
from nn import *
from util import *

matplotlib.use("Agg")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", default="data/toy", help="Input directory")
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=-1,
        help="GPU ID (negative value indicates CPU)",
    )
    parser.add_argument("-u", "--unit", type=int, default=300, help="Number of units")
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=32,
        help="Number of examples in each mini-batch",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        default=100,
        help="Max number of sweeps over the dataset to train",
    )
    parser.add_argument(
        "-p",
        "--patients",
        type=int,
        default=10,
        help="Early stopping trigger patient count",
    )
    parser.add_argument(
        "-o", "--out", default="data/toy/asc2v", help="Directory to output the result"
    )
    parser.add_argument(
        "-r", "--resume", default="", help="Resume the training from snapshot"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="result/model.params",
        help="Model output params file (for resume training)",
    )
    parser.add_argument(
        "-c",
        "--context",
        default="asc2v",
        choices=[
            "c2v",
            "asc2v",
        ],
    )
    parser.add_argument(
        "-t",
        "--trimfreq",
        default=0,
        type=int,
        help="minimum frequency for word in training",
    )
    parser.add_argument(
        "-a", "--alpha", type=float, default=0.99, help="RMSprop exponential decay rate"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.001,
        help="Initial learning rate for RMSprop",
    )
    parser.add_argument(
        "-mr",
        "--min_learning_rate",
        type=float,
        default=1e-8,
        help="Minimum learning rate",
    )
    parser.add_argument(
        "-sl", "--schedule_lr", action="store_true", help="Scheduling learning rate"
    )
    parser.add_argument("-ss", "--stepsize", type=int, default=5, help="Step size")
    parser.add_argument(
        "-bs",
        "--begin_step",
        type=int,
        default=0,
        help="Begin step size with initial learning rate",
    )
    parser.add_argument(
        "-lr_reduce",
        "--lr_reduce",
        type=float,
        default=0.1,
        help="Custom adaptive learning rate reduce factor",
    )
    parser.add_argument(
        "-gc",
        "--grad_clip",
        default=None,
        type=float,
        help="if specified, clip gradient l2 to this value",
    )
    parser.add_argument(
        "-np", "--ns_power", default=0.75, type=float, help="Negative sampling power"
    )
    parser.add_argument("-do", "--dropout", default=0.0, type=float, help="NN dropout")
    parser.add_argument(
        "-rs", "--random_seed", type=int, default=None, help="Random seed value"
    )
    parser.add_argument(
        "-opt",
        "--optimizer",
        default="rmsprop",
        choices=["adam", "rmsprop"],
        help="Optimizer",
    )
    parser.add_argument(
        "--amsgrad", action="store_true", help="Whether to use AMSGrad variant of Adam"
    )
    args = parser.parse_args()
    print("GPU:", args.gpu)
    print("# unit:", args.unit)
    print("# Minibatch-size:", args.batchsize)
    print("# epoch:", args.epoch)
    print("Seed value:", args.random_seed)
    print("Dropout:", args.dropout)
    print("Trimfreq:", args.trimfreq)
    print("NS power:", args.ns_power)
    print("Context:", args.context)
    print("Input directory:", args.indir)
    print("")
    return args


def schedule_optimizer_value(
    epoch_list, value_list, optimizer_name="main", attr_name="lr"
):
    """Set optimizer's hyperparameter according to value_list,
    scheduled on epoch_list.

    Example usage:
    trainer.extend(schedule_optimizer_value([2, 4, 7], [0.008, 0.006, 0.002]))
    """
    if isinstance(epoch_list, list):
        assert len(epoch_list) == len(value_list)
    else:
        assert isinstance(epoch_list, float) or isinstance(epoch_list, int)
        assert isinstance(value_list, float) or isinstance(value_list, int)
        epoch_list = [
            epoch_list,
        ]
        value_list = [
            value_list,
        ]

    trigger = triggers.ManualScheduleTrigger(epoch_list, "epoch")
    count = 0

    @chainer.training.extension.make_extension(trigger=trigger)
    def set_value(trainer: Trainer):
        nonlocal count
        optimizer = trainer.updater.get_optimizer(optimizer_name)
        setattr(optimizer, attr_name, value_list[count])
        count += 1

    return set_value


def convert(batch, device):
    converted = ()
    if device >= 0:
        if isinstance(batch, tuple):
            for data in batch:
                converted = converted + (cuda.to_gpu(data),)
        else:
            converted = cuda.to_gpu(batch)
    else:
        converted = batch
    return converted


def export_params(
    args,
    user2index,
    item2index,
    word2count,
    word2index,
    aspect2index,
    opinion2index,
    aspect_opinions,
):
    save_count(word2count, os.path.join(args.out, SC_WORD_COUNTS_FILENAME))
    save_dict(user2index, os.path.join(args.out, USER_DICT_FILENAME))
    save_dict(item2index, os.path.join(args.out, ITEM_DICT_FILENAME))
    save_dict(word2index, os.path.join(args.out, VOCAB_FILENAME))
    save_dict(aspect2index, os.path.join(args.out, SC_ASPECT_DICT_FILENAME))
    save_dict(opinion2index, os.path.join(args.out, SC_OPINION_DICT_FILENAME))
    dump_json(aspect_opinions, os.path.join(args.out, ASPECT_OPINION_FILENAME))
    file_path = "{}.params".format(os.path.join(args.out, MODEL_FILENAME))
    with open(file_path, "w") as f:
        f.write("model_filename\t{}\n".format(MODEL_FILENAME))
        f.write("model_type\t{}\n".format(args.context))
        f.write("unit\t{}\n".format(args.unit))
        f.write("ns_power\t{}\n".format(args.ns_power))
        f.write("user_filename\t{}\n".format(USER_DICT_FILENAME))
        f.write("item_filename\t{}\n".format(ITEM_DICT_FILENAME))
        f.write("vocab_filename\t{}\n".format(VOCAB_FILENAME))
        f.write("aspect_filename\t{}\n".format(SC_ASPECT_DICT_FILENAME))
        f.write("opinion_filename\t{}\n".format(SC_OPINION_DICT_FILENAME))
        f.write("aspect_opinions_filename\t{}\n".format(ASPECT_OPINION_FILENAME))
        f.write("#\t{}\n".format(" ".join(sys.argv)))


def get_dataset_iterator(context, data_loader, batchsize):
    val_file = os.path.join(data_loader.path, VALIDATION_FILENAME)
    val_data = data_loader.get_data(val_file)
    if context in ["c2v"]:
        train_iter = SentenceDynamicIterator(data_loader, batchsize, True)
        val_iter = SentenceIterator(val_data, batchsize, False)
    elif context in [
        "ac2v",
        "sc2v",
        "asc2v",
        "sc2v-mter",
        "asc2v-mter",
        "aoc2v",
        "rasc2v",
        "c2vas",
    ]:
        train_iter = SentimentSentenceDynamicIterator(
            context, data_loader, batchsize, True
        )
        val_iter = SentimentSentenceIterator(context, val_data, batchsize, False)
    return train_iter, val_iter


def get_context_model(args, data_loader):
    if args.resume:
        model_reader = ModelReader(args.resume, args.gpu, True, data_loader.word2count)
        model = model_reader.model
    else:
        n_vocab = data_loader.n_vocab
        if args.context in ["sc2v", "sc2v-mter"]:
            n_aspect = 1
        else:
            n_aspect = data_loader.n_aspect
        context_word_units = args.unit
        lstm_hidden_units = IN_TO_OUT_UNITS_RATIO * args.unit
        target_word_units = IN_TO_OUT_UNITS_RATIO * args.unit
        cs = [data_loader.word2count[w] for w in range(n_vocab)]
        loss_func = L.NegativeSampling(
            target_word_units, cs, NEGATIVE_SAMPLING_NUM, args.ns_power
        )
        loss_func.W.data[...] = 0
        if args.context == "c2v":
            model = Context2Vec(
                args.gpu,
                n_vocab,
                context_word_units,
                lstm_hidden_units,
                target_word_units,
                loss_func,
                True,
                args.dropout,
            )
        elif args.context in [
            "ac2v",
            "sc2v",
            "asc2v",
            "sc2v-mter",
            "asc2v-mter",
            "aoc2v",
            "rasc2v",
        ]:
            model = AspectSentiContext2Vec(
                args.gpu,
                n_vocab,
                n_aspect,
                context_word_units,
                lstm_hidden_units,
                target_word_units,
                loss_func,
                True,
                args.dropout,
            )
        elif args.context in ["c2vas"]:
            model = Context2VecAspectSenti(
                args.gpu,
                n_vocab,
                n_aspect,
                context_word_units,
                lstm_hidden_units,
                target_word_units,
                loss_func,
                True,
                args.dropout,
            )
    return model


def train(args):
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
    if args.random_seed:
        set_random_seed(args.random_seed, (args.gpu,))

    user2index = load_dict(os.path.join(args.indir, USER_DICT_FILENAME))
    item2index = load_dict(os.path.join(args.indir, ITEM_DICT_FILENAME))
    (trimmed_word2count, word2index, aspect2index, opinion2index) = read_and_trim_vocab(
        args.indir, args.trimfreq
    )
    aspect_opinions = get_aspect_opinions(os.path.join(args.indir, TRAIN_FILENAME))

    export_params(
        args,
        user2index,
        item2index,
        trimmed_word2count,
        word2index,
        aspect2index,
        opinion2index,
        aspect_opinions,
    )

    src_aspect_score = SOURCE_ASPECT_SCORE.get(args.context, "aspect_score_efm")

    data_loader = DataLoader(
        args.indir,
        user2index,
        item2index,
        trimmed_word2count,
        word2index,
        aspect2index,
        opinion2index,
        aspect_opinions,
        src_aspect_score,
    )

    train_iter, val_iter = get_dataset_iterator(
        args.context, data_loader, args.batchsize
    )

    model = get_context_model(args, data_loader)

    if args.optimizer == "rmsprop":
        optimizer = O.RMSprop(lr=args.learning_rate, alpha=args.alpha)
    elif args.optimizer == "adam":
        optimizer = O.Adam(amsgrad=args.amsgrad)

    optimizer.setup(model)
    if args.grad_clip:
        optimizer.add_hook(GradientClipping(args.grad_clip))
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu
    )
    early_stop = triggers.EarlyStoppingTrigger(
        monitor="validation/main/loss",
        patients=args.patients,
        max_trigger=(args.epoch, "epoch"),
    )
    trainer = training.Trainer(updater, stop_trigger=early_stop, out=args.out)
    trainer.extend(
        extensions.Evaluator(val_iter, model, converter=convert, device=args.gpu)
    )
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport(
            ["epoch", "main/loss", "validation/main/loss", "lr", "elapsed_time"]
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ["main/loss", "validation/main/loss"], x_key="epoch", file_name="loss.png"
        )
    )
    trainer.extend(extensions.ProgressBar())
    trainer.extend(
        extensions.snapshot_object(model, MODEL_FILENAME),
        trigger=triggers.MinValueTrigger("validation/main/loss"),
    )
    trainer.extend(extensions.observe_lr())

    if args.optimizer in ["rmsprop"]:
        if args.schedule_lr:
            epoch_list = np.array(
                [i for i in range(1, int(args.epoch / args.stepsize) + 1)]
            ).astype(np.int32)
            value_list = args.learning_rate * args.lr_reduce ** epoch_list
            value_list[value_list < args.min_learning_rate] = args.min_learning_rate
            epoch_list *= args.stepsize
            epoch_list += args.begin_step
            trainer.extend(
                schedule_optimizer_value(epoch_list.tolist(), value_list.tolist())
            )

    trainer.run()


if __name__ == "__main__":
    train(parse_arguments())
