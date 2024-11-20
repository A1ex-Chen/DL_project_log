def get_adapter_name(model):
    from peft.tuners.tuners_utils import BaseTunerLayer
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            return f'default_{len(module.r)}'
    return 'default_0'
