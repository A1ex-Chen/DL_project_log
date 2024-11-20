@register_model
def vit_so400m_patch14_siglip_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1152, depth=27, num_heads=16,
        mlp_ratio=3.7362, class_token=False, global_pool='map')
    model = _create_vision_transformer('vit_so400m_patch14_siglip_224',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
