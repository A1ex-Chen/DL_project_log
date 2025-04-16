def walk_objects(mod):
    already_seen = set()
    try:
        for _, modname, _ in pkgutil.walk_packages(mod.__path__, mod.
            __name__ + '.'):
            submod = importlib.import_module(modname)
            yield from walk_module_objects(submod, already_seen)
    except AttributeError:
        yield from walk_module_objects(mod, already_seen)
