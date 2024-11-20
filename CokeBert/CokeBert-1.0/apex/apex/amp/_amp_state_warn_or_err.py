def warn_or_err(msg):
    if _amp_state.hard_override:
        print('Warning:  ' + msg)
    else:
        raise RuntimeError(msg)
