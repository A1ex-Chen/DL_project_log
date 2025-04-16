def __init__(self, loss_scale='dynamic', enable_caching=True, verbose=False):
    self._enable_caching = enable_caching
    self._verbose = verbose
    self._cache = dict()
    self._default_scaler = LossScaler(loss_scale)
    self._is_active = True
    self._all_wrappers = []
