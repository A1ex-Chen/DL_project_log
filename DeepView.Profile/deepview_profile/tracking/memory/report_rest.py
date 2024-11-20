import collections
import enum

from deepview_profile.tracking.base import ReportBase, ReportBuilderBase
import deepview_profile.tracking.memory.report_queries as queries


WeightEntry = collections.namedtuple(
    'WeightEntry',
    ['weight_name',
     'size_bytes',
     'grad_size_bytes',
     'file_path',
     'line_number'],
)


ActivationEntry = collections.namedtuple(
    'ActivationEntry',
    ['operation_name', 'size_bytes', 'file_path', 'line_number'],
)


class MiscSizeType(enum.Enum):
    PeakUsageBytes = 'peak_usage_bytes'


class MemoryReport(ReportBase):





class MemoryReportBuilder(ReportBuilderBase):
    # This is the memory tracking report file format version that will be
    # created by this builder. When changes are made to the file format, this
    # integer should be increased monotonically.
    #
    # We need to version these tracking reports to protect us from future
    # changes to the file format.
    Version = 1








        cursor.executemany(queries.add_stack_frame, stack_frame_generator())