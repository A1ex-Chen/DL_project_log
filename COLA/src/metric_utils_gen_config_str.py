def gen_config_str(res):
    d, f, s, q, c, e = res
    sf = lambda s: '\\textbf{' + s + '}'
    on = lambda s, lb: sf(lb) if s else ''
    cfg_str = (
        f"{on(d, 'D')}{on(f, 'F')}{on(s and not d, 'S')}{on(q and not d, 'Q')}{on(c, 'C')}{on(e and not c, 'E')}"
        )
    if len(cfg_str) == 0:
        cfg_str = '$\\emptyset$'
    else:
        cfg_str = '+' + cfg_str
    return f' {cfg_str} '
