def measure_throughput(self):
    if self._profiler is None:
        self._initialize_iteration_profiler()
    num_samples = 3
    samples = self._profiler.sample_run_time_ms_by_batch_size(start_batch_size
        =self._batch_size, memory_usage_percentage=self.
        _memory_usage_percentage, start_batch_size_run_time_ms=self.
        _batch_size_iteration_run_time_ms, num_samples=num_samples)
    if len(samples) == 0 or samples[0].batch_size != self._batch_size:
        raise AnalysisError(
            "Something went wrong with DeepView.Profile when measuring your model's throughput. Please file a bug."
            )
    logger.debug('sampling results \n %r' % str(samples))
    measured_throughput = samples[0].batch_size / samples[0].run_time_ms * 1000
    throughput = pm.ThroughputResponse()
    throughput.samples_per_second = measured_throughput
    throughput.predicted_max_samples_per_second = math.nan
    throughput.can_manipulate_batch_size = False
    batch_info = self._entry_point_static_analyzer.batch_size_location()
    if batch_info is not None:
        throughput.batch_size_context.line_number = batch_info[0]
        throughput.can_manipulate_batch_size = batch_info[1]
        throughput.batch_size_context.file_path.components.extend(self.
            _entry_point.split(os.sep))
    if len(samples) != num_samples:
        return throughput
    batches = list(map(lambda sample: sample.batch_size, samples))
    run_times = list(map(lambda sample: sample.run_time_ms, samples))
    usages = list(map(lambda sample: sample.peak_usage_bytes, samples))
    run_time_model = _fit_linear_model(batches, run_times)
    peak_usage_model = _fit_linear_model(batches, usages)
    logger.debug('Run time model - Slope: %f, Bias: %f (ms)', *run_time_model)
    logger.debug('Peak usage model - Slope: %f, Bias: %f (bytes)', *
        peak_usage_model)
    throughput.peak_usage_bytes.slope = peak_usage_model[0]
    throughput.peak_usage_bytes.bias = peak_usage_model[1]
    predicted_max_throughput = 1000.0 / run_time_model[0]
    if run_time_model[0
        ] < 0.001 or measured_throughput > predicted_max_throughput:
        return throughput
    throughput.predicted_max_samples_per_second = predicted_max_throughput
    throughput.run_time_ms.slope = run_time_model[0]
    throughput.run_time_ms.bias = run_time_model[1]
    return throughput
