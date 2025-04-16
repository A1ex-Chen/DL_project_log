@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Example:
        ```python
        with temporary_modules({'old.module': 'new.module'}, {'old.module.attribute': 'new.module.attribute'}):
            import old.module  # this will now import new.module
            from old.module import attribute  # this will now import new.module.attribute
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module
    try:
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit('.', 1)
            new_module, new_attr = new.rsplit('.', 1)
            setattr(import_module(old_module), old_attr, getattr(
                import_module(new_module), new_attr))
        for old, new in modules.items():
            sys.modules[old] = import_module(new)
        yield
    finally:
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]
