@register_model
def vit_large_patch14_clip_quickgelu_336(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Large model (ViT-L/14) CLIP image tower @ 336x336 w/ QuickGELU act
    """
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16,
        pre_norm=True, norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer('vit_large_patch14_clip_quickgelu_336',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
