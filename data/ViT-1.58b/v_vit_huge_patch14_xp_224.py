@register_model
def vit_huge_patch14_xp_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Huge model (ViT-H/14) w/ parallel blocks and qk norm enabled.
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        pre_norm=True, no_embed_class=True, norm_layer=RmsNorm, block_fn=
        ParallelScalingBlock, qkv_bias=False, qk_norm=True)
    model = _create_vision_transformer('vit_huge_patch14_xp_224',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
