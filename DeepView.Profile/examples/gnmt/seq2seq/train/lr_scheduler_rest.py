import logging
import math

import torch





class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with exponential warmup and step decay.
    """
