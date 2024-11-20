import collections

from deepview_profile.tracking.base import ReportBase, ReportBuilderBase
import deepview_profile.tracking.time.report_queries as queries

RunTimeEntry = collections.namedtuple(
    'RunTimeEntry',
    ['operation_name',
     'forward_ms',
     'backward_ms',
     'file_path',
     'line_number'],
)


class OperationRunTimeReport(ReportBase):



class OperationRunTimeReportBuilder(ReportBuilderBase):
    # This is the operation run time tracking report file format version that
    # will be created by this builder. When changes are made to the file
    # format, this integer should be increased monotonically.
    #
    # We need to version these tracking reports to protect us from future
    # changes to the file format.
    Version = 1





        cursor.executemany(queries.add_stack_frame, stack_frame_generator())

    def build(self):
        self._connection.commit()
        return OperationRunTimeReport(self._connection)

    def _create_report_tables(self):
        cursor = self._connection.cursor()
        cursor.execute(queries.set_report_format_version.format(
            version=OperationRunTimeReportBuilder.Version))
        for creation_query in queries.create_report_tables.values():
            cursor.execute(creation_query)
        self._connection.commit()