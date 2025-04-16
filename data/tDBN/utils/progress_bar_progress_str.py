def progress_str(val, *args, width=20, with_ptg=True):
    val = max(0.0, min(val, 1.0))
    assert width > 1
    pos = round(width * val) - 1
    if with_ptg is True:
        log = '[{}%]'.format(max_point_str(val * 100.0, 4))
    log += '['
    for i in range(width):
        if i < pos:
            log += '='
        elif i == pos:
            log += '>'
        else:
            log += '.'
    log += ']'
    for arg in args:
        log += '[{}]'.format(arg)
    return log
