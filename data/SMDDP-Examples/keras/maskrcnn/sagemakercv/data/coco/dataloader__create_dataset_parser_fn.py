def _create_dataset_parser_fn(self):
    """Create parser for parsing input data (dictionary)."""
    return functools.partial(dataset_parser, mode=self._mode, params=self.
        _params, use_instance_mask=self._use_instance_mask, seed=self._seed)
