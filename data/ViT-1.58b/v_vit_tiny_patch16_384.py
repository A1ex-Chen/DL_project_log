@register_model
def vit_tiny_patch16_384(pretrained: bool=False, **kwargs) ->VisionTransformer:
    """ ViT-Tiny (Vit-Ti/16) @ 384x384.
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer('vit_tiny_patch16_384', pretrained=
        pretrained, **dict(model_args, **kwargs))
    return model
