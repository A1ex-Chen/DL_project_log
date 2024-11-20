def _check_callbacks(self, callbacks):
    if callbacks:
        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise ValueError(
                    'All callbacks must be instances of Callback class.')
