@contextlib.contextmanager
def user_code_environment(script_root_path, project_root):
    """
    A combined context manager that activates all relevant context managers
    used when running user code.
    """
    with sys_path_root(script_root_path):
        with exceptions_as_analysis_errors(project_root):
            yield
