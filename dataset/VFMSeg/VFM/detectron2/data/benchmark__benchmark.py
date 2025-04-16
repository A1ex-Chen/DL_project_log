def _benchmark(self, iterator, num_iter, warmup, msg=None):
    avg, all_times = iter_benchmark(iterator, num_iter, warmup, self.
        max_time_seconds)
    if msg is not None:
        self._log_time(msg, avg, all_times)
    return avg, all_times
