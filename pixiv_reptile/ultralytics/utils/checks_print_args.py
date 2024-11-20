def print_args(args: Optional[dict]=None, show_file=True, show_func=False):
    """Print function arguments (optional args dict)."""

    def strip_auth(v):
        """Clean longer Ultralytics HUB URLs by stripping potential authentication information."""
        return clean_url(v) if isinstance(v, str) and v.startswith('http'
            ) and len(v) > 100 else v
    x = inspect.currentframe().f_back
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={strip_auth(v)}' for k, v in
        args.items()))
