def text_encoder_mlp_modules(text_encoder):
    mlp_modules = []
    if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
        for i, layer in enumerate(text_encoder.text_model.encoder.layers):
            mlp_mod = layer.mlp
            name = f'text_model.encoder.layers.{i}.mlp'
            mlp_modules.append((name, mlp_mod))
    else:
        raise ValueError(
            f'do not know how to get mlp modules for: {text_encoder.__class__.__name__}'
            )
    return mlp_modules
