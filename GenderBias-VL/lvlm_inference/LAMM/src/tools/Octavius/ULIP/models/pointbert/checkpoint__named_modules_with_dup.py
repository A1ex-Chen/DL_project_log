def _named_modules_with_dup(model: nn.Module, prefix: str='') ->Iterable[Tuple
    [str, nn.Module]]:
    """
    The same as `model.named_modules()`, except that it includes
    duplicated modules that have more than one name.
    """
    yield prefix, model
    for name, module in model._modules.items():
        if module is None:
            continue
        submodule_prefix = prefix + ('.' if prefix else '') + name
        yield from _named_modules_with_dup(module, submodule_prefix)
