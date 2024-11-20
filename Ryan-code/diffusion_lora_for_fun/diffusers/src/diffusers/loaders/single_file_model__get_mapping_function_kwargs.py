def _get_mapping_function_kwargs(mapping_fn, **kwargs):
    parameters = inspect.signature(mapping_fn).parameters
    mapping_kwargs = {}
    for parameter in parameters:
        if parameter in kwargs:
            mapping_kwargs[parameter] = kwargs[parameter]
    return mapping_kwargs
