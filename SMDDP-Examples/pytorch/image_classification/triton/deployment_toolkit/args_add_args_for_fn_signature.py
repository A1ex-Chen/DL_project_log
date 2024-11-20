def add_args_for_fn_signature(parser, fn) ->argparse.ArgumentParser:
    parser.conflict_handler = 'resolve'
    signature = inspect.signature(fn)
    for parameter in signature.parameters.values():
        if parameter.name in ['self', 'args', 'kwargs']:
            continue
        argument_kwargs = {}
        if parameter.annotation != inspect.Parameter.empty:
            if parameter.annotation == bool:
                argument_kwargs['type'] = str2bool
                argument_kwargs['choices'] = [0, 1]
            elif isinstance(parameter.annotation, type(Optional[Any])):
                types = [type_ for type_ in parameter.annotation.__args__ if
                    not isinstance(None, type_)]
                if len(types) != 1:
                    raise RuntimeError(
                        f'Could not prepare argument parser for {parameter.name}: {parameter.annotation} in {fn}'
                        )
                argument_kwargs['type'] = types[0]
            else:
                argument_kwargs['type'] = parameter.annotation
        if parameter.default != inspect.Parameter.empty:
            if parameter.annotation == bool:
                argument_kwargs['default'] = str2bool(parameter.default)
            else:
                argument_kwargs['default'] = parameter.default
        else:
            argument_kwargs['required'] = True
        name = parameter.name.replace('_', '-')
        LOGGER.debug(f'Adding argument {name} with {argument_kwargs}')
        parser.add_argument(f'--{name}', **argument_kwargs)
    return parser
