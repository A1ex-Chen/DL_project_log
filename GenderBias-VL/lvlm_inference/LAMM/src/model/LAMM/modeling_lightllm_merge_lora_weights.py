def merge_lora_weights(lora_weight_path, lora_config: LoraConfig,
    pre_post_layer=None, transformer_layer_list: List[
    TransformerLayerWeight]=None):
    use_safetensors = lora_weight_path.endswith('.safetensors')
    if use_safetensors:
        lora_weights = safe_open(lora_weight_path, 'pt', 'cpu')
        lora_weights = {k: lora_weights.get_tensor(k) for k in lora_weights
            .keys()}
    else:
        lora_weights = torch.load(lora_weight_path, 'cpu')
        if pre_post_layer is not None:
            pass
        if transformer_layer_list is not None:
            for layer in transformer_layer_list:
                transformer_layer_load_qkvo(layer, lora_weights, lora_config)
    return
