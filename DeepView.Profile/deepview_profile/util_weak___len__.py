def __len__(self):
    if self._dirty_len and self._pending_removals:
        self._scrub_removals()
    return len(self.data) - len(self._pending_removals)
