def length_penalty(self):
    return ((5.0 + (self._step + 1)) / 6.0) ** self.alpha
