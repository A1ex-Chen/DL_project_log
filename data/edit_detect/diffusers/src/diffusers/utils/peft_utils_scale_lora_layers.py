def scale_lora_layers(model, weight):
    """
    Adjust the weightage given to the LoRA layers of the model.

    Args:
        model (`torch.nn.Module`):
            The model to scale.
        weight (`float`):
            The weight to be given to the LoRA layers.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer
    if weight == 1.0:
        return
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.scale_layer(weight)
