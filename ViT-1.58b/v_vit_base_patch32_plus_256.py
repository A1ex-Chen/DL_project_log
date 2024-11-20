@register_model
def vit_base_patch32_plus_256(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Base (ViT-B/32+)
    """
    model_args = dict(patch_size=32, embed_dim=896, depth=12, num_heads=14,
        init_values=1e-05)
    model = _create_vision_transformer('vit_base_patch32_plus_256',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
