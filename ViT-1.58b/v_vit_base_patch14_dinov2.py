@register_model
def vit_base_patch14_dinov2(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-B/14 for DINOv2
    """
    model_args = dict(patch_size=14, embed_dim=768, depth=12, num_heads=12,
        init_values=1e-05, img_size=518)
    model = _create_vision_transformer('vit_base_patch14_dinov2',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
