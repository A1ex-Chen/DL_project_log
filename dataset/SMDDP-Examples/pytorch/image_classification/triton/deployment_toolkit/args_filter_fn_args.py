def filter_fn_args(args: Union[dict, argparse.Namespace], fn: Callable) ->dict:
    signature = inspect.signature(fn)
    parameters_names = list(signature.parameters)
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    args = {k: v for k, v in args.items() if k in parameters_names}
    return args
