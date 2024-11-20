@register_model
def vit_giant_patch14_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Giant (little-g) model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_args = dict(patch_size=14, embed_dim=1408, mlp_ratio=48 / 11,
        depth=40, num_heads=16)
    model = _create_vision_transformer('vit_giant_patch14_224', pretrained=
        pretrained, **dict(model_args, **kwargs))
    return model
