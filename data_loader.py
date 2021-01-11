from util import to_one_hot
from defs import SENT_COUNTS_FILENAME, UNK_TOKEN
from tqdm import tqdm
import os
import csv
import math
import numpy as np

DATA_KEYS = [
    "sentences",
    "scores",
    "aspects",
    "aspects_w",
    "aspects_pos",
    "aspect1hot",
    "aspect_senti",
    "uias_vec",
    "random_aspect_senti",
    "opinions",
    "opinions_pos",
    "opinions_w",
    "ratings",
]


class DataLoader:
    sent_counts_filename = SENT_COUNTS_FILENAME

    def __init__(
        self,
        path,
        user2index,
        item2index,
        word2count,
        word2index,
        aspect2index,
        opinion2index,
        aspect_opinions={},
        src_aspect_score="aspect_score_efm",
    ):
        self.path = path
        self.user2index = user2index
        self.item2index = item2index
        self.word2count = word2count
        self.total_words = sum(word2count.values())
        self.word2index = word2index
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.aspect2index = aspect2index
        self.opinion2index = aspect2index
        self.n_user = max(self.user2index.values()) + 1
        self.n_item = max(self.item2index.values()) + 1
        self.n_vocab = max(self.word2index.values()) + 1
        self.n_aspect = max(self.aspect2index.values()) + 1
        self.n_opinion = max(self.opinion2index.values()) + 1
        self.aspect_opinions = aspect_opinions
        self.src_aspect_score = src_aspect_score
        self.fds = []

    def data_group_by_sentence_len(self, data_file_path):
        data_by_sentence_len = {}
        with open(data_file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in tqdm(
                reader, desc="Read data from {} into memory".format(data_file_path)
            ):
                data = data_by_sentence_len.setdefault(
                    row["sentence_len"], {key: [] for key in DATA_KEYS}
                )

                sent_words = row["sentence"].split()
                assert len(sent_words) > 1
                sent_inds = [
                    self.word2index.get(word, self.word2index[UNK_TOKEN])
                    for word in sent_words
                ]
                score = (
                    float(row[self.src_aspect_score])
                    if len(row[self.src_aspect_score].strip()) > 0
                    else 0.0
                )
                aspect_index = self.aspect2index.get(
                    row["aspect"], self.aspect2index[UNK_TOKEN]
                )
                aspect1hot = to_one_hot(aspect_index, self.n_aspect)
                user1hot = to_one_hot(self.user2index[row["reviewerID"]], self.n_user)
                item1hot = to_one_hot(self.item2index[row["asin"]], self.n_item)
                data["sentences"].append(sent_inds)
                data["scores"].append([score])
                data["aspects"].append(row["aspect"])
                data["aspects_pos"].append(row["aspect_pos"])
                aspect_w = self.word2index.get(
                    row["aspect"], self.word2index[UNK_TOKEN]
                )
                data["aspects_w"].append(aspect_w)
                data["aspect1hot"].append(aspect1hot)
                data["aspect_senti"].append(aspect1hot * score)
                data["uias_vec"].append(
                    np.concatenate((user1hot, item1hot, aspect1hot * score))
                )
                data["random_aspect_senti"].append(
                    to_one_hot(
                        np.random.choice(self.n_aspect),
                        self.n_aspect,
                        np.random.random(1).astype(np.float32) * 4 + 1,
                    )
                )  # score range from 1 to 5)
                data["opinions"].append(row["opinion"])
                data["opinions_pos"].append(row["opinion_pos"])
                opinion_w = self.word2index.get(
                    row["opinion"], self.word2index[UNK_TOKEN]
                )
                data["opinions_w"].append(opinion_w)
                data["ratings"].append(row["overall"])
        return data_by_sentence_len

    def get_data(self, data_file_path):
        return self.data_group_by_sentence_len(data_file_path)

    def open(self, batchsize=100):
        self.fds = []
        with open(os.path.join(self.path, self.sent_counts_filename)) as f:
            for line in f:
                [filename, count] = line.strip().split()
                batches = int(math.ceil(float(count) / batchsize))
                fd = open(os.path.join(self.path, filename), "r")
                reader = csv.DictReader(fd)
                self.fds = self.fds + [reader] * batches
        np.random.shuffle(self.fds)

    def read_batch(self, reader, batchsize, word2index):
        batch = {key: [] for key in DATA_KEYS}
        for row in reader:
            if len(batch["sentences"]) >= batchsize:
                break
            sent_words = row["sentence"].split()
            assert len(sent_words) > 1
            sent_inds = [
                word2index.get(word, word2index[UNK_TOKEN]) for word in sent_words
            ]
            score = (
                float(row[self.src_aspect_score])
                if len(row[self.src_aspect_score].strip()) > 0
                else 0.0
            )
            aspect_index = self.aspect2index.get(
                row["aspect"], self.aspect2index[UNK_TOKEN]
            )
            aspect1hot = to_one_hot(aspect_index, self.n_aspect)
            user1hot = to_one_hot(self.user2index[row["reviewerID"]], self.n_user)
            item1hot = to_one_hot(self.item2index[row["asin"]], self.n_item)
            batch["sentences"].append(sent_inds)
            batch["scores"].append([score])
            batch["aspects"].append([row["aspect"]])
            batch["aspects_pos"].append([row["aspect_pos"]])
            aspect_w = word2index.get(row["aspect"], word2index[UNK_TOKEN])
            batch["aspects_w"].append(aspect_w)
            batch["aspect1hot"].append(aspect1hot)
            batch["aspect_senti"].append(aspect1hot * score)
            batch["uias_vec"].append(
                np.concatenate((user1hot, item1hot, aspect1hot * score))
            )
            batch["random_aspect_senti"].append(
                to_one_hot(
                    np.random.choice(self.n_aspect),
                    self.n_aspect,
                    np.random.random(1).astype(np.float32) * 4 + 1,
                )
            )  # score range from 1 to 5)
            batch["opinions"].append(row["opinion"])
            batch["opinions_pos"].append(row["opinion_pos"])
            opinion_w = word2index.get(row["opinion"], word2index[UNK_TOKEN])
            batch["opinions_w"].append(opinion_w)
            batch["ratings"].append(row["overall"])
        return batch

    def batch_iter(self, batchsize=100):
        for fd in self.fds:
            batch = self.read_batch(fd, batchsize, self.word2index)
            if len(batch["sentences"]) > 0:
                yield batch
        yield {}
