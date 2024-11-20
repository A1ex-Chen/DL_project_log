def __iter__(self):
    self._req_thread.start()
    timeout_s = 0.05
    while True:
        try:
            ids, x, y_pred, y_real = self._results.get(timeout=timeout_s)
            yield ids, x, y_pred, y_real
        except queue.Empty:
            shall_stop = self._processed_all or self._errors
            if shall_stop:
                break
    LOGGER.debug('Waiting for request thread to stop')
    self._req_thread.join()
    if self._errors:
        error_msg = '\n'.join(map(str, self._errors))
        raise RuntimeError(error_msg)
