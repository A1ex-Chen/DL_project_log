@register_model
def vit_large_patch14_clip_quickgelu_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Large model (ViT-L/14) CLIP image tower w/ QuickGELU act
    """
    from timm.layers import get_act_layer
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16,
        pre_norm=True, norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer('vit_large_patch14_clip_quickgelu_224',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
