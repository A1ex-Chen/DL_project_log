def _remove_args(slices: List[Any]) ->None:
    """
    Used by get_perfetto_object to remove unused args
    Inputs: List of slices
    Outputs: None
    """
    [slice.pop('args', None) for slice in slices]
