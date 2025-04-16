def get_submodule(module, submodule_name: str) ->nn.Module:
    if submodule_name == '':
        return module
    err_msg = 'Cannot retrieve submodule {target} from the model: '
    for submodule in submodule_name.split('.'):
        if not hasattr(module, submodule):
            err_msg += module._get_name(
                ) + ' has no attribute `' + submodule + '`'
            raise AttributeError(err_msg)
        module = getattr(module, submodule)
        if not isinstance(module, nn.Module):
            err_msg += '`' + submodule + '` is not an nn.Module'
            raise AttributeError(err_msg)
    return module
