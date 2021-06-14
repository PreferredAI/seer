import math

import chainer
import chainer.initializers as I
from chainer import Variable
from chainer import functions as F
from chainer import links as L
from chainer import reporter

from defs import Toks


class Context2Vec(chainer.Chain):
    """
    Context2Vec model
    """

    def __init__(
        self,
        gpu,
        n_vocab,
        n_units,
        hidden_units,
        out_units,
        loss_func,
        train=True,
        drop_ratio=0.0,
    ):
        super(Context2Vec, self).__init__()
        with self.init_scope():
            self.l2r_embed = L.EmbedID(n_vocab, n_units)
            self.r2l_embed = L.EmbedID(n_vocab, n_units)
            self.loss_func = loss_func
            self.l2r_1 = L.LSTM(n_units, hidden_units)
            self.r2l_1 = L.LSTM(n_units, hidden_units)
            self.l3 = L.Linear(2 * hidden_units, 2 * hidden_units)
            self.l4 = L.Linear(2 * hidden_units, out_units)

        if gpu >= 0:
            self.to_gpu()
        self.l2r_embed.W.data = self.xp.random.normal(
            0,
            math.sqrt(1.0 / self.l2r_embed.W.data.shape[0]),
            self.l2r_embed.W.data.shape,
        ).astype(self.xp.float32)
        self.r2l_embed.W.data = self.xp.random.normal(
            0,
            math.sqrt(1.0 / self.r2l_embed.W.data.shape[0]),
            self.r2l_embed.W.data.shape,
        ).astype(self.xp.float32)
        self.train = train
        self.drop_ratio = drop_ratio

    def __call__(self, sent):
        self.reset_state()
        loss = self._calculate_loss(sent)
        reporter.report({"loss": loss}, self)
        return loss

    def reset_state(self):
        self.l2r_1.reset_state()
        self.r2l_1.reset_state()

    def _contexts_rep(self, sent_arr):
        batch_size = len(sent_arr)

        bos = self.xp.full((batch_size, 1), Toks.BOS, dtype=self.xp.int32)
        eos = self.xp.full((batch_size, 1), Toks.EOS, dtype=self.xp.int32)

        l2r_sent = self.xp.hstack((bos, sent_arr))  # <bos> a b c
        r2l_sent = self.xp.hstack((eos, sent_arr[:, ::-1]))  # <eos> c b a

        l2r_sent_h = []
        # ignore the last word in the sentence
        for i in range(l2r_sent.shape[1] - 1):
            c = Variable(l2r_sent[:, i])
            e = self.l2r_embed(c)
            if self.drop_ratio > 0:
                with chainer.using_config("train", self.train):
                    e = F.dropout(e, ratio=self.drop_ratio)
            h = self.l2r_1(e)
            l2r_sent_h.append(h)

        r2l_sent_h = []
        # ignore the last word in the sentence
        for i in range(r2l_sent.shape[1] - 1):
            c = Variable(r2l_sent[:, i])
            e = self.r2l_embed(c)
            if self.drop_ratio > 0:
                with chainer.using_config("train", self.train):
                    e = F.dropout(e, ratio=self.drop_ratio)
            h = self.r2l_1(e)
            r2l_sent_h.append(h)

        r2l_sent_h.reverse()

        sent_y = []
        for l2r_h, r2l_h in zip(l2r_sent_h, r2l_sent_h):
            bi_h = F.concat((l2r_h, r2l_h))
            if self.drop_ratio > 0:
                with chainer.using_config("train", self.train):
                    h1 = F.relu(self.l3(F.dropout(bi_h, ratio=self.drop_ratio)))
                    y = self.l4(F.dropout(h1, ratio=self.drop_ratio))
            else:
                h1 = F.relu(self.l3(bi_h))
                y = self.l4(h1)
            sent_y.append(y)
        return sent_y

    def _calculate_loss(self, sent):
        sent_arr = self.xp.asarray(sent, dtype=self.xp.int32)
        sent_y = self._contexts_rep(sent_arr)
        sent_x = []
        for i in range(sent_arr.shape[1]):
            x = Variable(sent_arr[:, i])
            sent_x.append(x)

        accum_loss = None
        for y, x in zip(sent_y, sent_x):
            loss = self.loss_func(y, x) / (len(x) * len(sent_x))
            accum_loss = accum_loss + loss if accum_loss is not None else loss

        return accum_loss

    def get_context_vector(self, x):
        sent, position = x
        self.reset_state()
        sent_arr = self.xp.asarray(sent, dtype=self.xp.int32)
        position = self.xp.asarray(position, dtype=self.xp.int32).tolist()
        sent_y = self._contexts_rep(sent_arr)

        context_vec = []
        for p, i in zip(position, range(len(position))):
            context_vec.append(sent_y[p].data[i])

        return context_vec


class AspectSentiContext2Vec(chainer.Chain):
    """
    Sentiment context model
    """

    def __init__(
        self,
        gpu,
        n_vocab,
        n_aspect,
        n_units,
        hidden_units,
        out_units,
        loss_func,
        train=True,
        drop_ratio=0.0,
    ):
        super(AspectSentiContext2Vec, self).__init__()
        with self.init_scope():
            self.l2r_embed = L.EmbedID(n_vocab, n_units)
            self.r2l_embed = L.EmbedID(n_vocab, n_units)
            self.l1 = L.Linear(n_aspect, n_units)
            self.loss_func = loss_func
            self.l2r_1 = L.LSTM(n_units, hidden_units)
            self.r2l_1 = L.LSTM(n_units, hidden_units)
            self.l3 = L.Linear(2 * hidden_units, 2 * hidden_units)
            self.l4 = L.Linear(2 * hidden_units, out_units)

        if gpu >= 0:
            self.to_gpu()
        self.l2r_embed.W.data = self.xp.random.normal(
            0,
            math.sqrt(1.0 / self.l2r_embed.W.data.shape[0]),
            self.l2r_embed.W.data.shape,
        ).astype(self.xp.float32)
        self.r2l_embed.W.data = self.xp.random.normal(
            0,
            math.sqrt(1.0 / self.r2l_embed.W.data.shape[0]),
            self.r2l_embed.W.data.shape,
        ).astype(self.xp.float32)
        self.train = train
        self.drop_ratio = drop_ratio

    def __call__(self, sent, aspect_senti):
        self.reset_state()
        loss = self._calculate_loss(sent, aspect_senti)
        reporter.report({"loss": loss}, self)
        return loss

    def reset_state(self):
        self.l2r_1.reset_state()
        self.r2l_1.reset_state()

    def _contexts_rep(self, sent_arr, aspect_senti_arr):
        batch_size = len(sent_arr)

        bos = self.xp.full((batch_size, 1), Toks.BOS, dtype=self.xp.int32)
        eos = self.xp.full((batch_size, 1), Toks.EOS, dtype=self.xp.int32)

        l2r_sent = self.xp.hstack((bos, sent_arr))  # <bos> a b c
        r2l_sent = self.xp.hstack((eos, sent_arr[:, ::-1]))  # <eos> c b a

        aspect_senti_v = Variable(aspect_senti_arr)
        if self.drop_ratio > 0:
            with chainer.using_config("train", self.train):
                aspect_senti_h = F.relu(
                    self.l1(F.dropout(aspect_senti_v, ratio=self.drop_ratio))
                )
        else:
            aspect_senti_h = F.relu(self.l1(aspect_senti_v))

        self.l2r_1(aspect_senti_h)
        self.r2l_1(aspect_senti_h)

        l2r_sent_h = []
        # ignore the last word in the sentence
        for i in range(l2r_sent.shape[1] - 1):
            c = Variable(l2r_sent[:, i])
            e = self.l2r_embed(c)
            if self.drop_ratio > 0:
                with chainer.using_config("train", self.train):
                    e = F.dropout(e, ratio=self.drop_ratio)
            h = self.l2r_1(e)
            l2r_sent_h.append(h)

        r2l_sent_h = []
        # ignore the last word in the sentence
        for i in range(r2l_sent.shape[1] - 1):
            c = Variable(r2l_sent[:, i])
            e = self.r2l_embed(c)
            if self.drop_ratio > 0:
                with chainer.using_config("train", self.train):
                    e = F.dropout(e, ratio=self.drop_ratio)
            h = self.r2l_1(e)
            r2l_sent_h.append(h)

        r2l_sent_h.reverse()

        sent_y = []
        for l2r_h, r2l_h in zip(l2r_sent_h, r2l_sent_h):
            bi_h = F.concat((l2r_h, r2l_h))
            if self.drop_ratio > 0:
                with chainer.using_config("train", self.train):
                    h1 = F.relu(self.l3(F.dropout(bi_h, ratio=self.drop_ratio)))
                    y = self.l4(F.dropout(h1, ratio=self.drop_ratio))
            else:
                h1 = F.relu(self.l3(bi_h))
                y = self.l4(h1)
            sent_y.append(y)
        return sent_y

    def _calculate_loss(self, sent, aspect_senti):
        sent_arr = self.xp.asarray(sent, dtype=self.xp.int32)
        aspect_senti_arr = self.xp.asarray(aspect_senti, dtype=self.xp.float32)
        sent_y = self._contexts_rep(sent_arr, aspect_senti_arr)
        sent_x = []
        for i in range(sent_arr.shape[1]):
            x = Variable(sent_arr[:, i])
            sent_x.append(x)

        accum_loss = None
        for y, x in zip(sent_y, sent_x):
            loss = self.loss_func(y, x) / (len(x) * len(sent_x))
            accum_loss = accum_loss + loss if accum_loss is not None else loss

        return accum_loss

    def get_context_vector(self, x):
        sent, position, aspect_senti = x
        self.reset_state()
        sent_arr = self.xp.asarray(sent, dtype=self.xp.int32)
        position = self.xp.asarray(position, dtype=self.xp.int32).tolist()
        aspect_senti_arr = self.xp.asarray(aspect_senti, dtype=self.xp.float32)
        sent_y = self._contexts_rep(sent_arr, aspect_senti_arr)

        context_vec = []
        for p, i in zip(position, range(len(position))):
            context_vec.append(sent_y[p].data[i])

        return context_vec
