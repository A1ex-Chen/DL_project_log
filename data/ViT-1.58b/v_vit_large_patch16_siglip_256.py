@register_model
def vit_large_patch16_siglip_256(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        class_token=False, global_pool='map')
    model = _create_vision_transformer('vit_large_patch16_siglip_256',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
