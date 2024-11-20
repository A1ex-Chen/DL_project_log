@register_model
def vit_large_patch14_reg4_dinov2(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-L/14 for DINOv2 w/ 4 registers
    """
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16,
        init_values=1e-05, reg_tokens=4, no_embed_class=True)
    model = _create_vision_transformer('vit_large_patch14_reg4_dinov2',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
