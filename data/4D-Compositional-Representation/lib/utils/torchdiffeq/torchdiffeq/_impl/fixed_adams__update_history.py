def _update_history(self, t, f):
    if self.prev_t is None or self.prev_t != t:
        self.prev_f.appendleft(f)
        self.prev_t = t
