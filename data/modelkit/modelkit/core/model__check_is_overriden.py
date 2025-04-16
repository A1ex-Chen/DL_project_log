def _check_is_overriden(self):
    if not hasattr(self._predict, '__not_overriden__'):
        self._predict_mode = PredictMode.SINGLE
    if not hasattr(self._predict_batch, '__not_overriden__'):
        if self._predict_mode == PredictMode.SINGLE:
            raise BothPredictsOverridenError(
                '_predict OR _predict_batch must be overriden, not both')
        self._predict_mode = PredictMode.BATCH
    if not self._predict_mode:
        raise NoPredictOverridenError(
            '_predict or _predict_batch must be overriden')
