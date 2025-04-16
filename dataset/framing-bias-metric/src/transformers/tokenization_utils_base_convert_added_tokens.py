def convert_added_tokens(obj: Union[AddedToken, Any], add_type_field=True):
    if isinstance(obj, AddedToken):
        out = obj.__getstate__()
        if add_type_field:
            out['__type'] = 'AddedToken'
        return out
    elif isinstance(obj, (list, tuple)):
        return list(convert_added_tokens(o, add_type_field=add_type_field) for
            o in obj)
    elif isinstance(obj, dict):
        return {k: convert_added_tokens(v, add_type_field=add_type_field) for
            k, v in obj.items()}
    return obj
