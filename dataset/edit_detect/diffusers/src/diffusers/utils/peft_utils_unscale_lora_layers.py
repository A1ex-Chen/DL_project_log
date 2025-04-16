def unscale_lora_layers(model, weight: Optional[float]=None):
    """
    Removes the previously passed weight given to the LoRA layers of the model.

    Args:
        model (`torch.nn.Module`):
            The model to scale.
        weight (`float`, *optional*):
            The weight to be given to the LoRA layers. If no scale is passed the scale of the lora layer will be
            re-initialized to the correct value. If 0.0 is passed, we will re-initialize the scale with the correct
            value.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer
    if weight == 1.0:
        return
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            if weight is not None and weight != 0:
                module.unscale_layer(weight)
            elif weight is not None and weight == 0:
                for adapter_name in module.active_adapters:
                    module.set_scale(adapter_name, 1.0)
