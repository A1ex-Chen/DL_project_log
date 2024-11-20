def progress_bar(self, iterable=None, total=None):
    if not hasattr(self, '_progress_bar_config'):
        self._progress_bar_config = {}
    elif not isinstance(self._progress_bar_config, dict):
        raise ValueError(
            f'`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}.'
            )
    if iterable is not None:
        return tqdm(iterable, **self._progress_bar_config)
    elif total is not None:
        return tqdm(total=total, **self._progress_bar_config)
    else:
        raise ValueError('Either `total` or `iterable` has to be defined.')
