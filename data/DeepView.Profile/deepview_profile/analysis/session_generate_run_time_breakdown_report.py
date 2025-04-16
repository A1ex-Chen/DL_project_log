def generate_run_time_breakdown_report(self, save_report_to):
    tracker = self._get_tracker_instance()
    tracker.track_run_time()
    tracker.get_run_time_report(report_file=save_report_to)
