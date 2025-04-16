def _process_preds(self, preds: list[Any]) ->np.ndarray:
    _processed_preds = []
    for pred in preds:
        _processed_preds.append(self._process_pred(pred))
    preds = np.array(_processed_preds)
    if not preds.size:
        preds = np.zeros((0, 6))
    return preds
