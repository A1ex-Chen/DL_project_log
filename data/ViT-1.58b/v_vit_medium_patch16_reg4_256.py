@register_model
def vit_medium_patch16_reg4_256(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=512, depth=12, num_heads=8,
        class_token=True, no_embed_class=True, reg_tokens=4)
    model = _create_vision_transformer('vit_medium_patch16_reg4_256',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
