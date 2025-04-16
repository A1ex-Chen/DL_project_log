def recurse_remove_peft_layers(model):
    """
    Recursively replace all instances of `LoraLayer` with corresponding new layers in `model`.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer
    has_base_layer_pattern = False
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            has_base_layer_pattern = hasattr(module, 'base_layer')
            break
    if has_base_layer_pattern:
        from peft.utils import _get_submodules
        key_list = [key for key, _ in model.named_modules() if 'lora' not in
            key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(model, key)
            except AttributeError:
                continue
            if hasattr(target, 'base_layer'):
                setattr(parent, target_name, target.get_base_layer())
    else:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                recurse_remove_peft_layers(module)
            module_replaced = False
            if isinstance(module, LoraLayer) and isinstance(module, torch.
                nn.Linear):
                new_module = torch.nn.Linear(module.in_features, module.
                    out_features, bias=module.bias is not None).to(module.
                    weight.device)
                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias
                module_replaced = True
            elif isinstance(module, LoraLayer) and isinstance(module, torch
                .nn.Conv2d):
                new_module = torch.nn.Conv2d(module.in_channels, module.
                    out_channels, module.kernel_size, module.stride, module
                    .padding, module.dilation, module.groups).to(module.
                    weight.device)
                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias
                module_replaced = True
            if module_replaced:
                setattr(model, name, new_module)
                del module
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return model
