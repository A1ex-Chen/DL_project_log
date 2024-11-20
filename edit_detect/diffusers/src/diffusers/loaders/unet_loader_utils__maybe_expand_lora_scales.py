def _maybe_expand_lora_scales(unet: 'UNet2DConditionModel', weight_scales:
    List[Union[float, Dict]], default_scale=1.0):
    blocks_with_transformer = {'down': [i for i, block in enumerate(unet.
        down_blocks) if hasattr(block, 'attentions')], 'up': [i for i,
        block in enumerate(unet.up_blocks) if hasattr(block, 'attentions')]}
    transformer_per_block = {'down': unet.config.layers_per_block, 'up': 
        unet.config.layers_per_block + 1}
    expanded_weight_scales = [_maybe_expand_lora_scales_for_one_adapter(
        weight_for_adapter, blocks_with_transformer, transformer_per_block,
        unet.state_dict(), default_scale=default_scale) for
        weight_for_adapter in weight_scales]
    return expanded_weight_scales
