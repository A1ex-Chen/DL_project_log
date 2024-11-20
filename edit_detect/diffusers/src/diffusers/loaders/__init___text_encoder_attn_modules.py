def text_encoder_attn_modules(text_encoder):
    deprecate('text_encoder_attn_modules in `models`', '0.27.0',
        '`text_encoder_lora_state_dict` is deprecated and will be removed in 0.27.0. Make sure to retrieve the weights using `get_peft_model`. See https://huggingface.co/docs/peft/v0.6.2/en/quicktour#peftmodel for more information.'
        )
    from transformers import CLIPTextModel, CLIPTextModelWithProjection
    attn_modules = []
    if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
        for i, layer in enumerate(text_encoder.text_model.encoder.layers):
            name = f'text_model.encoder.layers.{i}.self_attn'
            mod = layer.self_attn
            attn_modules.append((name, mod))
    else:
        raise ValueError(
            f'do not know how to get attention modules for: {text_encoder.__class__.__name__}'
            )
    return attn_modules
