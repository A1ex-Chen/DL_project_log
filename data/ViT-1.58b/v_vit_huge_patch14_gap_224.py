@register_model
def vit_huge_patch14_gap_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Huge model (ViT-H/14) w/ no class token, avg pool
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer('vit_huge_patch14_gap_224',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
