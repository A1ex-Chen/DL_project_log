def _scrub_removals(self):
    d = self.data
    self._pending_removals = [k for k in self._pending_removals if k in d]
    self._dirty_len = False
