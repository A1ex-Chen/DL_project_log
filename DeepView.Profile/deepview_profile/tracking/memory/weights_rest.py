import torch
import inspect

from deepview_profile.tracking.base import TrackerBase
from deepview_profile.tracking.call_stack import CallStack
from deepview_profile.tracking.hook_manager import HookManager
from deepview_profile.tracking.utils import tensor_size_bytes
from deepview_profile.util_weak import WeakTensorKeyDictionary

class WeightsTracker(TrackerBase):





        return hook