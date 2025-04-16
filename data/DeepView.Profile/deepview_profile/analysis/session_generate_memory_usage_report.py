def generate_memory_usage_report(self, save_report_to):
    self._prepare_for_memory_profiling()
    tracker = self._get_tracker_instance()
    tracker.track_memory()
    tracker.get_memory_report(report_file=save_report_to)
