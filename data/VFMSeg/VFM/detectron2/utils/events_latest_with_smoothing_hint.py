def latest_with_smoothing_hint(self, window_size=20):
    """
        Similar to :meth:`latest`, but the returned values
        are either the un-smoothed original latest value,
        or a median of the given window_size,
        depend on whether the smoothing_hint is True.

        This provides a default behavior that other writers can use.
        """
    result = {}
    for k, (v, itr) in self._latest_scalars.items():
        result[k] = self._history[k].median(window_size
            ) if self._smoothing_hints[k] else v, itr
    return result
