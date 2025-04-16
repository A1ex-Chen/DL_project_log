def _check_and_update(key, value):
    assert value is not None
    if key in _known_status:
        if not _known_status[key] == value:
            raise RuntimeError(
                'Confilict status for {}, existing status {}, new status {}'
                .format(key, _known_status[key], value))
    _known_status[key] = value
