def text_encoder_attn_modules(text_encoder):
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
