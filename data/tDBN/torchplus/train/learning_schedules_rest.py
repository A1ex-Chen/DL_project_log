"""PyTorch edition of TensorFlow learning schedule in tensorflow object
detection API. 
"""
import numpy as np
from torch.optim.optimizer import Optimizer
class _LRSchedulerStep(object):

    """
    def get_lr(self):
        raise NotImplementedError
    """





class Constant(_LRSchedulerStep):



class ManualStepping(_LRSchedulerStep):
    """Pytorch edition of manual_stepping in tensorflow.
    DON'T SUPPORT PARAM GROUPS.
    """




class ExponentialDecayWithBurnin(_LRSchedulerStep):
    """Pytorch edition of manual_stepping in tensorflow.
    """




class ExponentialDecay(_LRSchedulerStep):



class CosineDecayWithWarmup(_LRSchedulerStep):
