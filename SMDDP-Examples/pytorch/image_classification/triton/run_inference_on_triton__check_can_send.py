def _check_can_send():
    return self._num_waiting_for < self._max_unresp_reqs
