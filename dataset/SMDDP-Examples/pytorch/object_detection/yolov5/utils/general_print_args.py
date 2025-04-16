def print_args(args: Optional[dict]=None, show_file=True, show_fcn=False):
    x = inspect.currentframe().f_back
    file, _, fcn, _, _ = inspect.getframeinfo(x)
    if args is None:
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    s = (f'{Path(file).stem}: ' if show_file else '') + (f'{fcn}: ' if
        show_fcn else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))
