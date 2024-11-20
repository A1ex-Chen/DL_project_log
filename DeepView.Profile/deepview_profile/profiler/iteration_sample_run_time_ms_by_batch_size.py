def sample_run_time_ms_by_batch_size(self, start_batch_size,
    start_batch_size_run_time_ms=None, start_batch_size_peak_usage_bytes=
    None, memory_usage_percentage=None, num_samples=3):
    samples = []
    if (start_batch_size_run_time_ms is None or 
        start_batch_size_peak_usage_bytes is None):
        start_run_time_ms, start_peak_usage_bytes, _ = (self.
            measure_run_time_ms(start_batch_size))
    else:
        start_run_time_ms = start_batch_size_run_time_ms
        start_peak_usage_bytes = start_batch_size_peak_usage_bytes
    samples.append(IterationSample(start_batch_size, start_run_time_ms,
        start_peak_usage_bytes))
    max_batch_size = (start_batch_size / memory_usage_percentage if 
        memory_usage_percentage is not None else start_batch_size + 100)
    if len(samples) < num_samples:
        samples.extend(self._sample_range(start_batch_size, max_batch_size,
            num_samples=num_samples - len(samples), is_increasing=True))
    if len(samples) < num_samples:
        samples.extend(self._sample_range(1, start_batch_size, num_samples=
            num_samples - len(samples), is_increasing=False))
    return samples
