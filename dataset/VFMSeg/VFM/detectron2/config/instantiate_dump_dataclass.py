def dump_dataclass(obj: Any):
    """
    Dump a dataclass recursively into a dict that can be later instantiated.

    Args:
        obj: a dataclass object

    Returns:
        dict
    """
    assert dataclasses.is_dataclass(obj) and not isinstance(obj, type
        ), 'dump_dataclass() requires an instance of a dataclass.'
    ret = {'_target_': _convert_target_to_string(type(obj))}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if dataclasses.is_dataclass(v):
            v = dump_dataclass(v)
        if isinstance(v, (list, tuple)):
            v = [(dump_dataclass(x) if dataclasses.is_dataclass(x) else x) for
                x in v]
        ret[f.name] = v
    return ret
