def close(self):
    if hasattr(self, '_writer'):
        self._writer.close()
