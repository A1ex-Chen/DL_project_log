def _predict_batch(self, items, **kwargs):
    """A generic _predict_batch that stacks and passes items to TensorFlow"""
    mask = [self._is_empty(item) for item in items]
    if all(mask):
        return self._rebuild_predictions_with_mask(mask, {})
    vects = {key: np.stack([item[key] for item, mask in zip(items, mask) if
        not mask], axis=0) for key in items[0]}
    predictions = self._tensorflow_predict(vects)
    return self._rebuild_predictions_with_mask(mask, predictions)
