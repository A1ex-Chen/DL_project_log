@register_model
def vit_base_patch16_gap_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Base (ViT-B/16) w/o class token, w/ avg-pool @ 224x224
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=16,
        class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer('vit_base_patch16_gap_224',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
