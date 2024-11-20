def _get_eta(self, storage) ->Optional[str]:
    if self._max_iter is None:
        return ''
    iteration = storage.iter
    try:
        eta_seconds = storage.history('time').median(1000) * (self.
            _max_iter - iteration - 1)
        storage.put_scalar('eta_seconds', eta_seconds, smoothing_hint=False)
        return str(datetime.timedelta(seconds=int(eta_seconds)))
    except KeyError:
        eta_string = None
        if self._last_write is not None:
            estimate_iter_time = (time.perf_counter() - self._last_write[1]
                ) / (iteration - self._last_write[0])
            eta_seconds = estimate_iter_time * (self._max_iter - iteration - 1)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        self._last_write = iteration, time.perf_counter()
        return eta_string
