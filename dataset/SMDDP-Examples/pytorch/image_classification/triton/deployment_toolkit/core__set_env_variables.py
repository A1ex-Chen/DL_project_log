def _set_env_variables(self) ->Dict[str, object]:
    """this method not remove values; fix it if needed"""
    to_set = {}
    old_values = {k: os.environ.pop(k, None) for k in to_set}
    os.environ.update(to_set)
    return old_values
