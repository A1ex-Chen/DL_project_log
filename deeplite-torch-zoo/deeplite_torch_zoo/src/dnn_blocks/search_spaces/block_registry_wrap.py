def wrap(obj):
    cls_name = name
    if kwargs:
        combinations = itertools.product(*list(kwargs.values()))
        for combination in combinations:
            cls_name = name
            if cls_name is None:
                cls_name = obj.__name__
            obj_kwargs = dict(zip(list(kwargs.keys()), combination))
            new_obj = partial(obj, **obj_kwargs)
            cls_name = cls_name, tuple(obj_kwargs.items())
            _register(cls_name, new_obj)
    else:
        if cls_name is None:
            cls_name = obj.__name__
        cls_name = cls_name, {}
        _register(cls_name, obj)
    return obj
