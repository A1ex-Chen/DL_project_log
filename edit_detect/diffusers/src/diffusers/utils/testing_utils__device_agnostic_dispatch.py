def _device_agnostic_dispatch(device: str, dispatch_table: Dict[str,
    Callable], *args, **kwargs):
    if device not in dispatch_table:
        return dispatch_table['default'](*args, **kwargs)
    fn = dispatch_table[device]
    if fn is None:
        return None
    return fn(*args, **kwargs)
