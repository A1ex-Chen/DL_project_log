@register_model
def vit_base_patch16_clip_quickgelu_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-B/16 CLIP image tower w/ QuickGELU act
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
        pre_norm=True, norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer('vit_base_patch16_clip_quickgelu_224',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
