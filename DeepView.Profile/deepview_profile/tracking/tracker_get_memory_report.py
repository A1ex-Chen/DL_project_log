def get_memory_report(self, report_file=None):
    if (self._weight_tracker is None or self._activations_tracker is None or
        self._peak_usage_bytes is None):
        raise RuntimeError('Memory tracking has not been performed yet.')
    return MemoryReportBuilder(report_file).process_tracker(self.
        _weight_tracker).process_tracker(self._activations_tracker
        ).add_misc_entry(MiscSizeType.PeakUsageBytes, self._peak_usage_bytes
        ).build()
