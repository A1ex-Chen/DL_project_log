def gain_db(self, gain):
    self._samples *= 10.0 ** (gain / 20.0)
