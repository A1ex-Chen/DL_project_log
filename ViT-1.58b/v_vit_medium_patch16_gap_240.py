@register_model
def vit_medium_patch16_gap_240(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Medium (ViT-M/16) w/o class token, w/ avg-pool @ 240x240
    """
    model_args = dict(patch_size=16, embed_dim=512, depth=12, num_heads=8,
        class_token=False, global_pool='avg', qkv_bias=False, init_values=
        1e-06, fc_norm=False)
    model = _create_vision_transformer('vit_medium_patch16_gap_240',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
