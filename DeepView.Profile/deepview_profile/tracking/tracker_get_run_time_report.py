def get_run_time_report(self, report_file=None):
    if self._operation_tracker is None:
        raise RuntimeError('Run time tracking has not been performed yet.')
    return OperationRunTimeReportBuilder(report_file).process_tracker(self.
        _operation_tracker).build()
