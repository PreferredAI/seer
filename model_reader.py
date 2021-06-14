import os

import chainer.links as L
import chainer.serializers as S
import numpy as np
from chainer.backends import cuda

from defs import (IN_TO_OUT_UNITS_RATIO, NEGATIVE_SAMPLING_NUM,
                  SOURCE_ASPECT_SCORE)
from nn import AspectSentiContext2Vec, Context2Vec
from util import load_dict, load_json


class ModelReader():
    def __init__(self, config_file, gpu=-1, resume=False, word2count=None):
        self.gpu = gpu
        self.resume = resume
        self.word2count = word2count
        self.xp = cuda.cupy if gpu >= 0 else np
        self.params = self.read_config_file(config_file)
        (self.user2index, self.item2index, self.w,
         self.word2index, self.aspect2index, self.opinion2index,
         self.aspect_opinions, self.model) = self.read_model(self.params)
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.src_aspect_score = SOURCE_ASPECT_SCORE[
            self.params['model_type']] if self.params['model_type'] in SOURCE_ASPECT_SCORE else 'aspect_score_efm'

    def read_config_file(self, filename):
        params = {}
        config_path = filename[:filename.rfind('/') + 1]
        params['config_path'] = config_path
        with open(filename, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    [param, value] = line.strip().split()
                    params[param] = value
        return params

    def read_model(self, params):
        user_file = os.path.join(params['config_path'],
                                 params['user_filename'])
        item_file = os.path.join(params['config_path'],
                                 params['item_filename'])
        vocab_file = os.path.join(params['config_path'],
                                  params['vocab_filename'])
        aspect_file = os.path.join(params['config_path'],
                                   params['aspect_filename'])
        opinion_file = os.path.join(params['config_path'],
                                    params['opinion_filename'])
        aspect_opinions_file = os.path.join(params['config_path'],
                                            params['aspect_opinions_filename'])
        model_file = os.path.join(params['config_path'],
                                  params['model_filename'])

        context_word_units = int(params['unit'])
        lstm_hidden_units = IN_TO_OUT_UNITS_RATIO * context_word_units
        target_word_units = IN_TO_OUT_UNITS_RATIO * context_word_units

        user2index = load_dict(user_file)
        item2index = load_dict(item_file)
        word2index = load_dict(vocab_file)
        aspect2index = load_dict(aspect_file)
        opinion2index = load_dict(opinion_file)
        aspect_opinions = load_json(aspect_opinions_file)

        n_user = max(user2index.values()) + 1
        n_item = max(item2index.values()) + 1
        n_vocab = max(word2index.values()) + 1
        n_aspect = max(aspect2index.values()) + 1

        n_encode = n_aspect

        # dummy word counts - not used for eval
        cs = [1 for _ in range(n_vocab)]
        # dummy loss func - not used for eval
        loss_func = L.NegativeSampling(
            target_word_units, cs, NEGATIVE_SAMPLING_NUM)

        if params['model_type'] == 'c2v':
            model = Context2Vec(self.gpu, n_vocab, context_word_units,
                                lstm_hidden_units, target_word_units, loss_func, self.resume)
        elif params['model_type'] in ['asc2v']:
            model = AspectSentiContext2Vec(self.gpu, n_vocab, n_encode, context_word_units,
                                           lstm_hidden_units, target_word_units, loss_func, self.resume)
        S.load_npz(model_file, model)
        w = model.loss_func.W.data
        return user2index, item2index, w, word2index, aspect2index, opinion2index, aspect_opinions, model
