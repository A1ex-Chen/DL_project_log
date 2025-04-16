def put_scalar(self, name, value, smoothing_hint=True):
    """
        Add a scalar `value` to the `HistoryBuffer` associated with `name`.

        Args:
            smoothing_hint (bool): a 'hint' on whether this scalar is noisy and should be
                smoothed when logged. The hint will be accessible through
                :meth:`EventStorage.smoothing_hints`.  A writer may ignore the hint
                and apply custom smoothing rule.

                It defaults to True because most scalars we save need to be smoothed to
                provide any useful signal.
        """
    name = self._current_prefix + name
    history = self._history[name]
    value = float(value)
    history.update(value, self._iter)
    self._latest_scalars[name] = value, self._iter
    existing_hint = self._smoothing_hints.get(name)
    if existing_hint is not None:
        assert existing_hint == smoothing_hint, 'Scalar {} was put with a different smoothing_hint!'.format(
            name)
    else:
        self._smoothing_hints[name] = smoothing_hint
