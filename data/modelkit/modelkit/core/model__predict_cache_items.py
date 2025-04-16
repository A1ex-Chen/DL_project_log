def _predict_cache_items(self, _step: int, cache_items: List[CacheItem],
    _callback: Optional[Callable[[int, List[ItemType], List[ReturnType]],
    None]]=None, **kwargs) ->Iterator[ReturnType]:
    batch = [self._validate(res.item, self._item_model, errors.
        ItemValidationException) for res in cache_items if res.missing]
    try:
        predictions = iter(self._predict_batch(batch, **kwargs))
    except BaseException as exc:
        raise errors.PredictionError(exc=exc) from exc
    current_predictions = []
    try:
        for cache_item in cache_items:
            if cache_item.missing:
                current_predictions.append(next(predictions))
                if (cache_item.cache_key and self.configuration_key and
                    self.cache and self.model_settings.get('cache_predictions')
                    ):
                    self.cache.set(cache_item.cache_key,
                        current_predictions[-1])
                yield self._validate(current_predictions[-1], self.
                    _return_model, errors.ReturnValueValidationException)
            else:
                current_predictions.append(cache_item.cache_value)
                yield self._validate(current_predictions[-1], self.
                    _return_model, errors.ReturnValueValidationException)
    except GeneratorExit:
        pass
    if _callback:
        _callback(_step, batch, current_predictions)
