def _convert_negative_tids_to_positive(slices: List[Any]) ->None:
    """
    Used by get_perfetto_object to change ids from neg to pos
    Inputs: List of slices
    Outputs: None
    """
    for slice in slices:
        if 'tid' in slice and isinstance(slice['tid'], int):
            slice['tid'] = abs(slice['tid'])
