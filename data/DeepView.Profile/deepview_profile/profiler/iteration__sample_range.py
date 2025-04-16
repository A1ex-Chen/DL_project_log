def _sample_range(self, min_size, max_size, num_samples, is_increasing=True):
    samples = []
    stack = [(min_size, max_size)]
    while len(samples) < num_samples and len(stack) > 0:
        lower, upper = stack.pop()
        if lower >= upper:
            continue
        next_size = self._select_batch_size(lower, upper, is_increasing)
        logger.debug('[%d, %d] Sampling batch size: %d', lower, upper,
            next_size)
        err, result = self.measure_run_time_ms_catch_oom(next_size)
        if err is not None:
            stack.append((lower, next_size - 1))
            continue
        samples.append(IterationSample(next_size, result[0], result[1]))
        if is_increasing:
            stack.append((lower, next_size - 1))
            stack.append((next_size + 1, upper))
        else:
            stack.append((next_size + 1, upper))
            stack.append((lower, next_size - 1))
    return samples
