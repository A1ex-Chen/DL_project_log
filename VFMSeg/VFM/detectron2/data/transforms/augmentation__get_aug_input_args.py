def _get_aug_input_args(aug, aug_input) ->List[Any]:
    """
    Get the arguments to be passed to ``aug.get_transform`` from the input ``aug_input``.
    """
    if aug.input_args is None:
        prms = list(inspect.signature(aug.get_transform).parameters.items())
        if len(prms) == 1:
            names = 'image',
        else:
            names = []
            for name, prm in prms:
                if prm.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.
                    Parameter.VAR_KEYWORD):
                    raise TypeError(
                        f' The default implementation of `{type(aug)}.__call__` does not allow `{type(aug)}.get_transform` to use variable-length arguments (*args, **kwargs)! If arguments are unknown, reimplement `__call__` instead. '
                        )
                names.append(name)
        aug.input_args = tuple(names)
    args = []
    for f in aug.input_args:
        try:
            args.append(getattr(aug_input, f))
        except AttributeError as e:
            raise AttributeError(
                f"{type(aug)}.get_transform needs input attribute '{f}', but it is not an attribute of {type(aug_input)}!"
                ) from e
    return args
