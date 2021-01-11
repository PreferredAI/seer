from builtins import *  # NOQA

import os
import random

import chainer
import numpy as np


def set_random_seed(seed, gpus=()):
    """Set a given random seed to ChainerRL's random sources.
    This function sets a given random seed to random sources that ChainerRL
    depends on so that ChainerRL can be deterministic. It is not responsible
    for setting a random seed to environments ChainerRL is applied to.
    Note that there's no guaranteed way to make all the computations done by
    Chainer deterministic. See https://github.com/chainer/chainer/issues/4134.
    Args:
      seed (int): Random seed [0, 2 ** 32).
      gpus (tuple of ints): GPU device IDs to use. Negative values are ignored.
    """
    # ChainerRL depends on random
    random.seed(seed)
    # ChainerRL depends on numpy.random
    np.random.seed(seed)
    # ChainerRL depends on cupy.random for GPU computation
    for gpu in gpus:
        if gpu >= 0:
            with chainer.cuda.get_device_from_id(gpu):
                chainer.cuda.cupy.random.seed(seed)
    # chainer.functions.n_step_rnn directly depends on CHAINER_SEED
    os.environ['CHAINER_SEED'] = str(seed)


if __name__ == '__main__':
    import sys
    random_seed, gpu = int(sys.argv[1]), int(sys.argv[2])
    if gpu >= 0:
        chainer.cuda.check_cuda_available()
        chainer.cuda.get_device(gpu).use()
    set_random_seed(random_seed, (gpu,))

    if gpu >= 0:
        print([chainer.cuda.cupy.random.randint(1000) for i in range(10)])
    print([np.random.randint(1000) for i in range(10)])
