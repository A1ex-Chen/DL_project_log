@register_model
def vit_small_patch32_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Small (ViT-S/32)
    """
    model_args = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch32_224', pretrained=
        pretrained, **dict(model_args, **kwargs))
    return model
