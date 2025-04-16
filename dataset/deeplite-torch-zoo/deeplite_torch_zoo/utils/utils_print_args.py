def print_args(args: Optional[dict]=None, show_file=True, show_func=False):
    x = inspect.currentframe().f_back
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    kv = ', '.join(f'{k}={v}' for k, v in args.items())
    LOGGER.info('%s', colorstr(s) + kv)
