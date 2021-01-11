import numpy as np
import chainer


class SentenceIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batchsize, repeat=True):
        self.dataset = dataset
        self.batchsize = batchsize
        self._repeat = repeat
        self.sent_lengths = list(dataset.keys())
        self.length_order = np.random.permutation(len(self.sent_lengths)).astype(
            np.int32
        )
        self.current_length_pos = 0
        self.current_df = np.array(
            self.dataset[self.sent_lengths[self.length_order[self.current_length_pos]]][
                "sentences"
            ]
        ).astype(np.int32)
        self.df_order = np.random.permutation(len(self.current_df)).astype(np.int32)
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        self.total_data_count = sum(
            [
                len(sentences_by_len["sentences"])
                for sentences_by_len in self.dataset.values()
            ]
        )
        self.current_data_count = 0

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batchsize
        position = self.df_order[i:i_end]
        next_batch = self.current_df.take(position, 0)
        if i_end >= len(self.df_order):
            self.current_length_pos += 1
            if self.current_length_pos >= len(self.length_order):
                np.random.shuffle(self.length_order)
                self.epoch += 1
                self.is_new_epoch = True
                self.current_length_pos = 0
                self.current_data_count = 0
            else:
                self.is_new_epoch = False
                self.current_data_count += len(position)
            self.current_df = np.array(
                self.dataset[
                    self.sent_lengths[self.length_order[self.current_length_pos]]
                ]["sentences"]
            ).astype(np.int32)
            self.df_order = np.random.permutation(len(self.current_df)).astype(np.int32)
            self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end
            self.current_data_count += self.batchsize

        return next_batch

    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_data_count) / self.total_data_count

    def serializer(self, serializer):
        self.current_data_count = serializer(
            "current_position", self.current_data_count
        )
        self.epoch = serializer("epoch", self.epoch)
        self.is_new_epoch = serializer("is_new_epoch", self.is_new_epoch)


class SentenceDynamicIterator(chainer.dataset.Iterator):
    def __init__(self, data_loader, batchsize, repeat=True):
        self.data_loader = data_loader
        self.batchsize = batchsize
        self.data_loader.open(batchsize)
        self._repeat = repeat
        self.epoch = 0
        self.is_new_epoch = False
        self.total_data_count = self.data_loader.total_words
        self.current_data_count = 0
        self.df = self.data_loader.batch_iter(self.batchsize)
        self.current_data = next(self.df)

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        sentences = np.array(self.current_data["sentences"]).astype(np.int32)
        self.current_data_count += len(sentences) * len(sentences[0])
        self.current_data = next(self.df)
        if not self.current_data:
            self.epoch += 1
            self.is_new_epoch = True
            self.current_data_count = 0
            self.data_loader.open(self.batchsize)
            self.df = self.data_loader.batch_iter(self.batchsize)
            self.current_data = next(self.df)
        else:
            self.is_new_epoch = False

        return sentences

    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_data_count) / self.total_data_count

    def serializer(self, serializer):
        self.current_data_count = serializer(
            "current_position", self.current_data_count
        )
        self.epoch = serializer("epoch", self.epoch)
        self.is_new_epoch = serializer("is_new_epoch", self.is_new_epoch)


class SentimentSentenceIterator(chainer.dataset.Iterator):
    def __init__(self, context, dataset, batchsize, repeat=True):
        self.context = context
        self.dataset = dataset
        self.batchsize = batchsize
        self._repeat = repeat
        self.sent_lengths = list(dataset.keys())
        self.length_order = np.random.permutation(len(self.sent_lengths)).astype(
            np.int32
        )
        self.current_length_pos = 0
        self.current_df = self.dataset[
            self.sent_lengths[self.length_order[self.current_length_pos]]
        ]
        self.current_df_sentences, self.current_df_sentitments = self.get_df()
        self.df_order = np.random.permutation(len(self.current_df_sentences)).astype(
            np.int32
        )
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        self.total_data_count = sum(
            [len(sub["sentences"]) for sub in self.dataset.values()]
        )
        self.current_data_count = 0

    def get_df(self):
        df_sentences = np.array(self.current_df["sentences"]).astype(np.int32)
        if self.context == "ac2v":
            df_sentiments = np.array(self.current_df["aspect1hot"]).astype(np.float32)
        elif self.context in ["sc2v", "sc2v-mter"]:
            df_sentiments = np.array(self.current_df["scores"]).astype(np.float32)
        elif self.context in ["asc2v", "asc2v-mter", "aoc2v", "c2vas"]:
            df_sentiments = np.array(self.current_df["aspect_senti"]).astype(np.float32)
        elif self.context in ["rasc2v"]:
            df_sentiments = np.array(self.current_df["random_aspect_senti"]).astype(
                np.float32
            )
        elif self.context in ["uiasc2v"]:
            df_sentiments = np.array(self.current_df["uias_vec"]).astype(np.float32)
        return df_sentences, df_sentiments

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batchsize
        position = self.df_order[i:i_end]
        sentences = self.current_df_sentences.take(position, 0)
        scores = self.current_df_sentitments.take(position, 0)
        if i_end >= len(self.df_order):
            self.current_length_pos += 1
            if self.current_length_pos >= len(self.length_order):
                np.random.shuffle(self.length_order)
                self.epoch += 1
                self.is_new_epoch = True
                self.current_length_pos = 0
                self.current_data_count = 0
            else:
                self.is_new_epoch = False
                self.current_data_count += len(position)
            self.current_df = self.dataset[
                self.sent_lengths[self.length_order[self.current_length_pos]]
            ]
            self.current_df_sentences, self.current_df_sentitments = self.get_df()

            self.df_order = np.random.permutation(
                len(self.current_df_sentences)
            ).astype(np.int32)
            self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end
            self.current_data_count += self.batchsize

        return sentences, scores

    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_data_count) / self.total_data_count

    def serializer(self, serializer):
        self.current_data_count = serializer(
            "current_position", self.current_data_count
        )
        self.epoch = serializer("epoch", self.epoch)
        self.is_new_epoch = serializer("is_new_epoch", self.is_new_epoch)


class SentimentSentenceDynamicIterator(chainer.dataset.Iterator):
    def __init__(self, context, data_loader, batchsize, repeat=True):
        self.context = context
        self.data_loader = data_loader
        self.batchsize = batchsize
        self.data_loader.open(batchsize)
        self._repeat = repeat
        self.epoch = 0
        self.is_new_epoch = False
        self.total_data_count = self.data_loader.total_words
        self.current_data_count = 0
        self.df = self.data_loader.batch_iter(self.batchsize)
        self.current_data = next(self.df)

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        sentences = np.array(self.current_data["sentences"]).astype(np.int32)
        if self.context == "ac2v":
            sentiments = np.array(self.current_data["aspect1hot"]).astype(np.float32)
        elif self.context in ["sc2v", "sc2v-mter"]:
            sentiments = np.array(self.current_data["scores"]).astype(np.float32)
        elif self.context in ["asc2v", "asc2v-mter", "aoc2v", "c2vas"]:
            sentiments = np.array(self.current_data["aspect_senti"]).astype(np.float32)
        elif self.context in ["rasc2v"]:
            sentiments = np.array(self.current_data["random_aspect_senti"]).astype(
                np.float32
            )
        elif self.context in ["uiasc2v"]:
            sentiments = np.array(self.current_data["uias_vec"]).astype(np.float32)
        self.current_data_count += len(sentences) * len(sentences[0])
        self.current_data = next(self.df)
        if not self.current_data:
            self.epoch += 1
            self.is_new_epoch = True
            self.current_data_count = 0
            self.data_loader.open(self.batchsize)
            self.df = self.data_loader.batch_iter(self.batchsize)
            self.current_data = next(self.df)
        else:
            self.is_new_epoch = False

        return sentences, sentiments

    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_data_count) / self.total_data_count

    def serializer(self, serializer):
        self.current_data_count = serializer(
            "current_position", self.current_data_count
        )
        self.epoch = serializer("epoch", self.epoch)
        self.is_new_epoch = serializer("is_new_epoch", self.is_new_epoch)


class DataIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batchsize, repeat=True):
        self.dataset = dataset
        self.batchsize = batchsize
        self._repeat = repeat
        self.sent_lengths = list(dataset.keys())
        self.length_order = np.random.permutation(len(self.sent_lengths)).astype(
            np.int32
        )
        self.current_length_pos = 0
        self.current_df = self.dataset[
            self.sent_lengths[self.length_order[self.current_length_pos]]
        ]
        self.dfs = self.get_dfs()
        self.df_order = np.random.permutation(len(self.dfs["sentences"])).astype(
            np.int32
        )
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        self.total_data_count = sum(
            [len(sub["sentences"]) for sub in self.dataset.values()]
        )
        self.current_data_count = 0
        self.current_batchsize = 0

    def get_dfs(self):
        dfs = {k: np.array(v) for k, v in self.current_df.items() if len(v) > 0}
        return dfs

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batchsize
        position = self.df_order[i:i_end]
        self.current_batchsize = len(position)

        batch = {k: v.take(position, 0) for k, v in self.dfs.items()}

        if i_end >= len(self.df_order):
            self.current_length_pos += 1
            if self.current_length_pos >= len(self.length_order):
                np.random.shuffle(self.length_order)
                self.epoch += 1
                self.is_new_epoch = True
                self.current_length_pos = 0
                self.current_data_count = 0
            else:
                self.is_new_epoch = False
            self.current_df = self.dataset[
                self.sent_lengths[self.length_order[self.current_length_pos]]
            ]
            self.dfs = self.get_dfs()

            self.df_order = np.random.permutation(len(self.dfs["sentences"])).astype(
                np.int32
            )
            self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end
        self.current_data_count += self.current_batchsize

        return batch

    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_data_count) / self.total_data_count

    def serializer(self, serializer):
        self.current_data_count = serializer(
            "current_position", self.current_data_count
        )
        self.epoch = serializer("epoch", self.epoch)
        self.is_new_epoch = serializer("is_new_epoch", self.is_new_epoch)
