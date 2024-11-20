def _reset_fields(self):
    """
        Before each uodate call, reset fields first
        """
    self._matched_idx = set()
    self._matched_ID = set()
    self._untracked_prev_idx = set(range(len(self._prev_instances)))
