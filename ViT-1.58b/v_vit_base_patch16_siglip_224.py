@register_model
def vit_base_patch16_siglip_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
        class_token=False, global_pool='map')
    model = _create_vision_transformer('vit_base_patch16_siglip_224',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
