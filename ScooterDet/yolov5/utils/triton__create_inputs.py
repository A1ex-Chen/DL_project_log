def _create_inputs(self, *args, **kwargs):
    args_len, kwargs_len = len(args), len(kwargs)
    if not args_len and not kwargs_len:
        raise RuntimeError('No inputs provided.')
    if args_len and kwargs_len:
        raise RuntimeError('Cannot specify args and kwargs at the same time')
    placeholders = self._create_input_placeholders_fn()
    if args_len:
        if args_len != len(placeholders):
            raise RuntimeError(
                f'Expected {len(placeholders)} inputs, got {args_len}.')
        for input, value in zip(placeholders, args):
            input.set_data_from_numpy(value.cpu().numpy())
    else:
        for input in placeholders:
            value = kwargs[input.name]
            input.set_data_from_numpy(value.cpu().numpy())
    return placeholders
