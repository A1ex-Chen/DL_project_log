import collections


from deepview_profile.tracking.call_stack import CallStack
from deepview_profile.tracking.base import TrackerBase
from deepview_profile.tracking.callable_tracker import CallableTracker
from deepview_profile.tracking.utils import remove_dunder
from deepview_profile.profiler.operation import OperationProfiler

OperationInfo = collections.namedtuple(
    'OperationInfo', ['operation_name', 'stack', 'forward_ms', 'backward_ms'])


class OperationRunTimeTracker(TrackerBase):






        return hook