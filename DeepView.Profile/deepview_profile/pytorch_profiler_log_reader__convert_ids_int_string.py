def _convert_ids_int_string(slices: List[Any]) ->None:
    """
    Used by get_perfetto_object to convert ids to str
    Inputs: List of slices
    Outputs: None
    """
    for slice in slices:
        if 'id' in slice:
            slice['id'] = str(slice['id'])
