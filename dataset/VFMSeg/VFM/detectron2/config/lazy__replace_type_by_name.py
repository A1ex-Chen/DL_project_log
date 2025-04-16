def _replace_type_by_name(x):
    if '_target_' in x and callable(x._target_):
        try:
            x._target_ = _convert_target_to_string(x._target_)
        except AttributeError:
            pass
