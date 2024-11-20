from bisect import bisect_right
from typing import List, Optional

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class WarmUpMultiStepLR(_LRScheduler):


