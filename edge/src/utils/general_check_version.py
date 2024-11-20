def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned
    =False, hard=False, verbose=False):
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = current == minimum if pinned else current >= minimum
    s = (
        f'WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'
        )
    if hard:
        assert result, emojis(s)
    if verbose and not result:
        LOGGER.warning(s)
    return result
