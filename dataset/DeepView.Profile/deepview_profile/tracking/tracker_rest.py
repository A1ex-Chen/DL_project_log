import enum
import logging

import torch

from deepview_profile.tracking.backward_interceptor import BackwardInterceptor
from deepview_profile.tracking.breakdown import HierarchicalBreakdownBuilder
from deepview_profile.tracking.memory.activations import ActivationsTracker
from deepview_profile.tracking.memory.report import MemoryReportBuilder, MiscSizeType
from deepview_profile.tracking.memory.weights import WeightsTracker
from deepview_profile.tracking.time.operation import OperationRunTimeTracker
from deepview_profile.tracking.time.report import OperationRunTimeReportBuilder
from deepview_profile.user_code_utils import user_code_environment

logger = logging.getLogger(__name__)


class Tracker:







class _TrackerState(enum.Enum):
    CREATED = 0
    MEMORY_TRACKED = 1
    RUN_TIME_TRACKED = 2