from sportslabkit.logger import logger
from sportslabkit.matching import SimpleMatchingFunction
from sportslabkit.metrics import IoUCMM
from sportslabkit.mot.base import MultiObjectTracker


class SORTTracker(MultiObjectTracker):
    """SORT tracker from https://arxiv.org/pdf/1602.00763.pdf"""

    hparam_search_space = {
        "max_staleness": {"type": "int", "low": 1, "high": 1e3},
        "min_length": {"type": "int", "low": 1, "high": 1e3},
    }



    @property

    @property