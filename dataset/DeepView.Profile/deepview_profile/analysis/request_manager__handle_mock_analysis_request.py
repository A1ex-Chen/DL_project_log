def _handle_mock_analysis_request(self, analysis_request, context):
    breakdown = pm.BreakdownResponse()
    breakdown.peak_usage_bytes = 1337
    breakdown.memory_capacity_bytes = 13337
    breakdown.iteration_run_time_ms = 133.7
    self._message_sender.send_breakdown_response(breakdown, context)
    throughput = pm.ThroughputResponse()
    throughput.samples_per_second = 1337
    throughput.predicted_max_samples_per_second = math.nan
    self._message_sender.send_throughput_response(throughput, context)
