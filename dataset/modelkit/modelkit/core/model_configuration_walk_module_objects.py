def walk_module_objects(mod, already_seen):
    for name, obj in inspect.getmembers(mod):
        if inspect.isclass(obj) and issubclass(obj, Asset) and name not in {
            'Model', 'Asset', 'TensorflowModel'} and obj not in already_seen:
            already_seen.add(obj)
            yield obj
