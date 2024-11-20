def get_objects_from_module(module):
    """
    Args:
    Returns a dict of object names and values in a module, while skipping private/internal objects
        module (ModuleType):
            Module to extract the objects from.

    Returns:
        dict: Dictionary of object names and corresponding values
    """
    objects = {}
    for name in dir(module):
        if name.startswith('_'):
            continue
        objects[name] = getattr(module, name)
    return objects
