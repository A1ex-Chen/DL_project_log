def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned
    =False, hard=False, verbose=False):
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = current == minimum if pinned else current >= minimum
    if hard:
        info = (
            f'⚠️ {name}{minimum} is required by YOLOv6, but {name}{current} is currently installed'
            )
        assert result, info
    return result
