@modelkit_predict_profiler
@errors.wrap_modelkit_exceptions_gen
def predict_gen(self, items: Iterator[ItemType], batch_size: Optional[int]=
    None, _callback: Optional[Callable[[int, List[ItemType], List[
    ReturnType]], None]]=None, _force_compute: bool=False, **kwargs
    ) ->Iterator[ReturnType]:
    batch_size = batch_size or (self.batch_size or 1)
    n_items_to_compute = 0
    n_items_from_cache = 0
    cache_items: List[CacheItem] = []
    step = 0
    for current_item in items:
        if self.configuration_key and self.cache and self.model_settings.get(
            'cache_predictions'):
            if not _force_compute:
                cache_item = self.cache.get(self.configuration_key,
                    current_item, kwargs)
            else:
                cache_item = CacheItem(current_item, self.cache.hash_key(
                    self.configuration_key, current_item, kwargs), None, True)
            if cache_item.missing:
                n_items_to_compute += 1
            else:
                n_items_from_cache += 1
            cache_items.append(cache_item)
        else:
            cache_items.append(CacheItem(current_item, None, None, True))
            n_items_to_compute += 1
        if batch_size and (n_items_to_compute == batch_size or 
            n_items_from_cache == 2 * batch_size):
            yield from self._predict_cache_items(step, cache_items,
                _callback=_callback, **kwargs)
            cache_items = []
            n_items_to_compute = 0
            n_items_from_cache = 0
            step += batch_size
    if cache_items:
        yield from self._predict_cache_items(step, cache_items, _callback=
            _callback, **kwargs)
