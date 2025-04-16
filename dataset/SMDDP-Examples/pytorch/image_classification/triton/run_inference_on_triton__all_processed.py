def _all_processed():
    LOGGER.debug(f'wait for {self._num_waiting_for} unprocessed jobs')
    return self._num_waiting_for == 0
