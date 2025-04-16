@register_model
def vit_base_patch16_xp_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Large model (ViT-L/14) w/ parallel blocks and qk norm enabled.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
        pre_norm=True, no_embed_class=True, norm_layer=RmsNorm, block_fn=
        ParallelScalingBlock, qkv_bias=False, qk_norm=True)
    model = _create_vision_transformer('vit_base_patch16_xp_224',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
