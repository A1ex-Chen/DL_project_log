def format_args(args):
    args_as_dict = asdict(args)
    args_as_dict = {k: (f'<{k.upper()}>' if k.endswith('_token') else v) for
        k, v in args_as_dict.items()}
    attrs_as_str = [f'{k}={v},' for k, v in sorted(args_as_dict.items())]
    return f"{args.__class__.__name__}\n({' '.join(attrs_as_str)})"
