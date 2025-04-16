def create_prior_lora_layers(unet: nn.Module):
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        lora_attn_processor_class = LoRAAttnProcessor2_0 if hasattr(F,
            'scaled_dot_product_attention') else LoRAAttnProcessor
        lora_attn_procs[name] = lora_attn_processor_class(hidden_size=unet.
            config.c)
    unet_lora_layers = AttnProcsLayers(lora_attn_procs)
    return lora_attn_procs, unet_lora_layers
