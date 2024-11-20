@register_model
def vit_giant_patch16_gap_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Giant (little-gg) model (ViT-g/16) w/ no class token, avg pool
    """
    model_args = dict(patch_size=16, embed_dim=1408, depth=40, num_heads=16,
        mlp_ratio=48 / 11, class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer('vit_giant_patch16_gap_224',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
