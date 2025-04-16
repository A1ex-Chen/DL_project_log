def unet_lora_state_dict(unet: UNet2DConditionModel) ->Dict[str, torch.Tensor]:
    """
    Returns:
        A state dict containing just the LoRA parameters.
    """
    lora_state_dict = {}
    for name, module in unet.named_modules():
        if hasattr(module, 'set_lora_layer'):
            lora_layer = getattr(module, 'lora_layer')
            if lora_layer is not None:
                current_lora_layer_sd = lora_layer.state_dict()
                for lora_layer_matrix_name, lora_param in current_lora_layer_sd.items(
                    ):
                    lora_state_dict[f'{name}.lora.{lora_layer_matrix_name}'
                        ] = lora_param
    return lora_state_dict
