from sportslabkit.logger import logger
from sportslabkit.matching import MotionVisualMatchingFunction
from sportslabkit.metrics import CosineCMM, IoUCMM
from sportslabkit.mot.base import MultiObjectTracker


class DeepSORTTracker(MultiObjectTracker):
    """DeepSORT tracker from https://arxiv.org/abs/1703.07402"""

    hparam_search_space = {
        "max_staleness": {"type": "int", "low": 1, "high": 1e3},
        "min_length": {"type": "int", "low": 1, "high": 1e3},
    }



    @property

    @property