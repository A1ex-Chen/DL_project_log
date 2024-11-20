def reset(self):
    for col in self._data.columns:
        self._data[col].values[:] = 0
