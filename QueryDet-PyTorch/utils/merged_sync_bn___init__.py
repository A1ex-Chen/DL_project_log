def __init__(self, *args, stats_mode='', **kwargs):
    super().__init__(*args, **kwargs)
    assert stats_mode in ['', 'N']
    self._stats_mode = stats_mode
    self._batch_mean = None
    self._batch_meansqr = None
