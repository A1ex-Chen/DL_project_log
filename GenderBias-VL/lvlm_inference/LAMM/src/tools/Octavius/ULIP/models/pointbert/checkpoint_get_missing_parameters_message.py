def get_missing_parameters_message(keys: List[str]) ->str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the model but not found in a checkpoint.
    Args:
        keys (list[str]): List of keys that were not found in the checkpoint.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = 'Some model parameters or buffers are not found in the checkpoint:\n'
    msg += '\n'.join('  ' + colored(k + _group_to_str(v), 'blue') for k, v in
        groups.items())
    return msg
