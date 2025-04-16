@contextlib.contextmanager
def sys_path_root(script_root_path):
    """
    A context manager that sets sys.path[0] to the specified path on entry and
    then restores it after exiting the context manager.
    """
    skyline_script_root = sys.path[0]
    try:
        sys.path[0] = script_root_path
        yield
    finally:
        sys.path[0] = skyline_script_root
