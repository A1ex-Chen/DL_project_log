def measure_breakdown(self, nvml):
    self._prepare_for_memory_profiling()
    tracker = self._get_tracker_instance()
    tracker.track_memory()
    tracker.track_run_time()
    breakdown = tracker.get_hierarchical_breakdown()
    del tracker
    if self._batch_size_iteration_run_time_ms is None:
        if self._profiler is None:
            self._initialize_iteration_profiler()
        (self._batch_size_iteration_run_time_ms, self.
            _batch_size_peak_usage_bytes, _
            ) = self._profiler.measure_run_time_ms(self._batch_size)
    bm = pm.BreakdownResponse()
    bm.batch_size = self._batch_size
    bm.peak_usage_bytes = breakdown.peak_usage_bytes
    bm.memory_capacity_bytes = nvml.get_memory_capacity().total
    bm.iteration_run_time_ms = self._batch_size_iteration_run_time_ms
    breakdown.operations.serialize_to_protobuf(bm.operation_tree)
    breakdown.weights.serialize_to_protobuf(bm.weight_tree)
    self._memory_usage_percentage = (bm.peak_usage_bytes / bm.
        memory_capacity_bytes)
    return bm
