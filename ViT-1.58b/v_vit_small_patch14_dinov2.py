@register_model
def vit_small_patch14_dinov2(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-S/14 for DINOv2
    """
    model_args = dict(patch_size=14, embed_dim=384, depth=12, num_heads=6,
        init_values=1e-05, img_size=518)
    model = _create_vision_transformer('vit_small_patch14_dinov2',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
