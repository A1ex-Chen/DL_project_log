def fix_one(qualname, name, obj):
    if id(obj) in seen_ids:
        return
    seen_ids.add(id(obj))
    mod = getattr(obj, '__module__', None)
    if mod is not None and (mod.startswith(module_name) or mod.startswith(
        'fvcore.')):
        obj.__module__ = module_name
        if hasattr(obj, '__name__') and '.' not in obj.__name__:
            obj.__name__ = name
            obj.__qualname__ = qualname
        if isinstance(obj, type):
            for attr_name, attr_value in obj.__dict__.items():
                fix_one(objname + '.' + attr_name, attr_name, attr_value)
