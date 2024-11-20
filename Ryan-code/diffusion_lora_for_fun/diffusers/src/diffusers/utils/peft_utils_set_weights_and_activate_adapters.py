def set_weights_and_activate_adapters(model, adapter_names, weights):
    from peft.tuners.tuners_utils import BaseTunerLayer

    def get_module_weight(weight_for_adapter, module_name):
        if not isinstance(weight_for_adapter, dict):
            return weight_for_adapter
        for layer_name, weight_ in weight_for_adapter.items():
            if layer_name in module_name:
                return weight_
        parts = module_name.split('.')
        key = f'{parts[0]}.{parts[1]}.attentions.{parts[3]}'
        block_weight = weight_for_adapter.get(key, 1.0)
        return block_weight
    for adapter_name, weight in zip(adapter_names, weights):
        for module_name, module in model.named_modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, 'set_adapter'):
                    module.set_adapter(adapter_name)
                else:
                    module.active_adapter = adapter_name
                module.set_scale(adapter_name, get_module_weight(weight,
                    module_name))
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            if hasattr(module, 'set_adapter'):
                module.set_adapter(adapter_names)
            else:
                module.active_adapter = adapter_names
