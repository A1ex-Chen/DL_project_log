from sportslabkit.logger import logger
from sportslabkit.matching import MotionVisualMatchingFunction, SimpleMatchingFunction
from sportslabkit.metrics import CosineCMM, IoUCMM
from sportslabkit.mot.base import MultiObjectTracker


class BYTETracker(MultiObjectTracker):
    """BYTE tracker from https://arxiv.org/pdf/2110.06864.pdf"""



    @property

    @property