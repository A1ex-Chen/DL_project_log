def set_adapter_layers(model, enabled=True):
    from peft.tuners.tuners_utils import BaseTunerLayer
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            if hasattr(module, 'enable_adapters'):
                module.enable_adapters(enabled=enabled)
            else:
                module.disable_adapters = not enabled
