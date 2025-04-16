@register_model
def vit_small_patch16_384(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Small (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch16_384', pretrained=
        pretrained, **dict(model_args, **kwargs))
    return model
