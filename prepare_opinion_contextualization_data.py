import argparse
import multiprocessing as mp
import os

import pandas as pd
from tqdm import tqdm

from defs import *
from efm import EFMReader
from util import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", type=str, default="data/toy")
    parser.add_argument(
        "-o", "--out", type=str, default="data/toy", help="Ouput directory"
    )
    parser.add_argument("-efm", "--efm_dir", type=str, default="data/toy/efm")
    parser.add_argument(
        "--max_sent_len", type=int, default=128, help="Max sentence length"
    )
    return parser.parse_args()


def export_text_completion_data(profile, efm, path):
    df = profile
    df["uif"] = df[["reviewerID", "asin", "aspect"]].apply(tuple, axis=1)
    df["aspect_score_efm"] = df.apply(
        lambda row: efm.get_aspect_score(row["reviewerID"], row["asin"], row["aspect"]),
        axis=1,
    )
    df = df.drop(columns=["uif"])
    df.to_csv(path, index=False)
    print("Done export data to", path)



def corpus_by_sent_length(corpus_dir, train_file, max_sent_len=128):
    df = pd.read_csv(train_file)
    df = df[df["sentence_len"] <= max_sent_len]
    sentence_lens = df["sentence_len"].unique()
    for l in tqdm(sentence_lens, desc='Grouping by sentence length'):
        df_l = df[df["sentence_len"] == l]
        df_l.to_csv(os.path.join(corpus_dir, "sent.{}".format(l)), index=False)

    save_count(
        list(df["sentence_len"].apply(lambda x: "sent.{}".format(x))),
        os.path.join(corpus_dir, SENT_COUNTS_FILENAME),
    )
    all_words = " ".join(df["sentence"].tolist()).split()
    save_count(all_words, os.path.join(corpus_dir, WORD_COUNTS_FILENAME))
    save_count(list(df["aspect"]), os.path.join(corpus_dir, ASPECT_COUNTS_FILENAME))
    save_count(list(df["opinion"]), os.path.join(corpus_dir, OPINION_COUNTS_FILENAME))

    with open(os.path.join(corpus_dir, TOTAL_COUNTS_FILENAME), "w") as totals_file:
        totals_file.write("total sentences read: {}\n".format(len(df)))
        totals_file.write("total words read: {}\n".format(len(all_words)))

def read_and_trim_vocab(corpus_dir, trimfreq):
    word2count = load_count(os.path.join(corpus_dir, WORD_COUNTS_FILENAME))
    aspect2count = load_count(os.path.join(corpus_dir, ASPECT_COUNTS_FILENAME))
    opinion2count = load_count(os.path.join(corpus_dir, OPINION_COUNTS_FILENAME))
    word2index = {"<UNK>": Toks.UNK, "<BOS>": Toks.BOS, "<EOS>": Toks.EOS}
    trimmed_word2count = Counter()
    trimmed_word2count[word2index["<UNK>"]] = 0
    trimmed_word2count[word2index["<BOS>"]] = 0
    trimmed_word2count[word2index["<EOS>"]] = 0
    for word, count in word2count.items():
        if count >= trimfreq and word.lower() != "<unk>" and word.lower() != "<rw>":
            ind = len(word2index)
            word2index[word] = ind
            trimmed_word2count[ind] = count
        else:
            trimmed_word2count[word2index["<UNK>"]] += count
    trimmed_aspects = [aspect for aspect in aspect2count if aspect in word2index]
    trimmed_opinions = [opinion for opinion in opinion2count if opinion in word2index]
    aspect2index = to_dict(["<UNK>"] + trimmed_aspects)
    opinion2index = to_dict(["<UNK>"] + trimmed_opinions)

    return trimmed_word2count, word2index, aspect2index, opinion2index


def get_aspect_opinions(path):
    aspect_opinions = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Getting aspect-opinion pairs at {}".format(path)):
            aspect, opinion = row["aspect"], row["opinion"]
            if aspect not in aspect_opinions:
                aspect_opinions[aspect] = []
            if opinion not in aspect_opinions[aspect]:
                aspect_opinions[aspect].append(opinion)
    return aspect_opinions

def main(args):
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    input_dir = args.indir
    train = pd.read_csv(os.path.join(input_dir, SPLITTED_TRAIN_FILE))
    validation = pd.read_csv(os.path.join(input_dir, SPLITTED_VALIDATION_FILE))
    # keep only opinions appear in training data
    validation = validation[validation["opinion"].isin(list(train["opinion"]))]
    test = pd.read_csv(os.path.join(input_dir, SPLITTED_TEST_FILE))
    # keep only opinions appear in training data
    test = test[test["opinion"].isin(list(train["opinion"]))]

    efm = EFMReader(args.efm_dir)
    mp.Process(
        target=export_text_completion_data,
        args=(test, efm, os.path.join(args.out, TEST_FILENAME)),
    ).start()
    mp.Process(
        target=export_text_completion_data,
        args=(validation, efm, os.path.join(args.out, VALIDATION_FILENAME)),
    ).start()
    t1 = mp.Process(
        target=export_text_completion_data,
        args=(train, efm, os.path.join(args.out, TRAIN_FILENAME)),
    )

    t1.start()
    t1.join()

    corpus_by_sent_length(args.out, os.path.join(args.out, TRAIN_FILENAME), args.max_sent_len)

if __name__ == "__main__":
    main(parse_args())
