def build_hook(self, args, hook_type=None):
    if isinstance(args, Hook):
        return args
    elif isinstance(args, dict):
        assert issubclass(hook_type, Hook)
        return hook_type(**args)
    else:
        raise TypeError('"args" must be either a Hook object or dict, not {}'
            .format(type(args)))
