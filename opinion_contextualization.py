import argparse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from model_reader import ModelReader
from util import to_one_hot, substitute_word
from efm import EFMReader


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='data/toy/test.csv',
                        help='Input file')
    parser.add_argument('-p', '--preference_dir', type=str,
                        default='data/toy/efm',
                        help='Base model for getting preference score')
    parser.add_argument('-m', '--model_path', type=str,
                        default='result/model.params',
                        help='ASC2V model file path')
    parser.add_argument('-o', '--out', type=str,
                        default='contextualized.csv', help='Output path')
    parser.add_argument('--debug', action='store_true',
                        help='Debug with smaller data')
    return parser.parse_args()


class OpinionContextualizer():
    def __init__(self, model_path, preference, strategy='', verbose=False):
        self.model_reader = ModelReader(model_path, -1)
        assert 'asc2v' in self.model_reader.params['model_type']
        opinion2index = self.model_reader.opinion2index
        self.word2index = self.model_reader.word2index
        self.aspect2index = self.model_reader.aspect2index
        self.aspect_opinions = self.model_reader.aspect_opinions
        self.strategy = strategy
        if 'aspect-opinion' not in self.strategy:
            self.candidates = [self.word2index[w]
                               for w in opinion2index.keys()
                               if w in self.word2index and w != '<UNK>']
            self.w_candidates = self.model_reader.w[self.candidates]
        self.index2word = {v: k
                           for k, v in self.word2index.items()}
        self.n_aspect = max(self.aspect2index.values()) + 1
        self.preference = preference
        self.verbose = verbose
        if self.verbose:
            print('Init OpinionContextualizer from %s' % model_path)

    def get_ranked_opinions(self, user, item, sentence, aspect_position,
                            opinion_position, top_k=None):
        sentence = sentence.split()
        aspect_position = int(float(aspect_position))
        opinion_position = int(float(opinion_position))
        aspect = sentence[aspect_position]
        if 'aspect-opinion' in self.strategy:
            self.candidates = [self.word2index[w]
                               for w in self.aspect_opinions[aspect]
                               if w in self.word2index and w != '<UNK>']
            self.w_candidates = self.model_reader.w[self.candidates]
        aspect_index = self.aspect2index.get(
            aspect, self.aspect2index['<UNK>'])
        aspect1hot = to_one_hot(aspect_index, self.n_aspect)
        score = self.preference.get_aspect_score(user, item, aspect)
        sentence_v = [self.word2index.get(word, self.word2index['<UNK>'])
                      for word in sentence]
        x = ([sentence_v], [opinion_position], [aspect1hot * score])
        similarity = cosine_similarity(
            self.model_reader.model.get_context_vector(x),
            self.w_candidates).reshape(len(self.candidates))
        ranked_ids = (-similarity).argsort()[0:top_k]
        ranked_candidates = np.array(self.candidates).take(ranked_ids)
        return [self.index2word[idx] for idx in ranked_candidates]


def contextualize(df, contextualizer, top_k=None, verbose=False):
    if len(df) > 0:
        df = df.copy()
        df['top k opinions'] = df.apply(
            lambda row: contextualizer.get_ranked_opinions(
                row['reviewerID'],
                row['asin'],
                row['sentence'],
                row['aspect_pos'],
                row['opinion_pos'], top_k=top_k), axis=1)

        df['predicted opinion'] = df['top k opinions'].apply(lambda x: x[0])

        df['original sentence'] = df['sentence']

        df['sentence'] = df.apply(
            lambda row: substitute_word(
                row['sentence'],
                row['predicted opinion'],
                row['opinion_pos']), axis=1)

    return df


if __name__ == '__main__':
    args = parse_arguments()
    df = pd.read_csv(args.input)
    if args.debug:
        df = df[0:100]
    preference = EFMReader(args.preference_dir)
    contextualizer = OpinionContextualizer(args.model_path, preference)
    df = contextualize(df, contextualizer)
    df.to_csv(args.out, index=False)
