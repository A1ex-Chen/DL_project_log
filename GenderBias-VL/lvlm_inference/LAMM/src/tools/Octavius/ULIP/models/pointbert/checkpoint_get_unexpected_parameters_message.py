def get_unexpected_parameters_message(keys: List[str]) ->str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the checkpoint but not found in the model.
    Args:
        keys (list[str]): List of keys that were not found in the model.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = (
        'The checkpoint state_dict contains keys that are not used by the model:\n'
        )
    msg += '\n'.join('  ' + colored(k + _group_to_str(v), 'magenta') for k,
        v in groups.items())
    return msg
