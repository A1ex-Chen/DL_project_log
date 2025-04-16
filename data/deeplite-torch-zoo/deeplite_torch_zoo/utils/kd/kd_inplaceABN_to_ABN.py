def inplaceABN_to_ABN(module: nn.Module) ->nn.Module:
    if isinstance(module, InplaceAbn):
        module_new = inplace_abn.ABN(module.num_features, activation=module
            .act_name, activation_param=module.act_param)
        for key in module.state_dict():
            module_new.state_dict()[key].copy_(module.state_dict()[key])
        module_new.training = module.training
        module_new.weight.data = module_new.weight.abs() + module_new.eps
        return module_new
    for name, child in reversed(module._modules.items()):
        new_child = inplaceABN_to_ABN(child)
        if new_child != child:
            module._modules[name] = new_child
    return module
