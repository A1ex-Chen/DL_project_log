from __future__ import division
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR


class WarmupMultiStepLR(_LRScheduler):
    """https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/solver/lr_scheduler.py"""




class ClipLR(object):
    """Clip the learning rate of a given scheduler.
    Same interfaces of _LRScheduler should be implemented.

    Args:
        scheduler (_LRScheduler): an instance of _LRScheduler.
        min_lr (float): minimum learning rate.

    """


