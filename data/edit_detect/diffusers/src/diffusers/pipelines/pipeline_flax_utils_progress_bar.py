def progress_bar(self, iterable):
    if not hasattr(self, '_progress_bar_config'):
        self._progress_bar_config = {}
    elif not isinstance(self._progress_bar_config, dict):
        raise ValueError(
            f'`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}.'
            )
    return tqdm(iterable, **self._progress_bar_config)
