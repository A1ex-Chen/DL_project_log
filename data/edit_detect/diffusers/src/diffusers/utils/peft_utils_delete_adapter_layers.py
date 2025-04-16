def delete_adapter_layers(model, adapter_name):
    from peft.tuners.tuners_utils import BaseTunerLayer
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            if hasattr(module, 'delete_adapter'):
                module.delete_adapter(adapter_name)
            else:
                raise ValueError(
                    'The version of PEFT you are using is not compatible, please use a version that is greater than 0.6.1'
                    )
    if getattr(model, '_hf_peft_config_loaded', False) and hasattr(model,
        'peft_config'):
        model.peft_config.pop(adapter_name, None)
        if len(model.peft_config) == 0:
            del model.peft_config
            model._hf_peft_config_loaded = None
