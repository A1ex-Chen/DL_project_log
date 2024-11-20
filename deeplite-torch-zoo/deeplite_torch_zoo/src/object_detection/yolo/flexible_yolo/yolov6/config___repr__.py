def __repr__(self):
    return 'Config (path: {}): {}'.format(self.filename, self._cfg_dict.
        __repr__())
