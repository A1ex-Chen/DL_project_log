@contextlib.contextmanager
def prevent_module_caching():
    """
    A context manager that prevents any imported modules from being cached
    after exiting.
    """
    try:
        original_modules = sys.modules.copy()
        yield
    finally:
        newly_added = {module_name for module_name in sys.modules.keys() if
            module_name not in original_modules}
        for module_name in newly_added:
            del sys.modules[module_name]
