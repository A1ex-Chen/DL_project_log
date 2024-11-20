@register_model
def vit_huge_patch14_clip_quickgelu_378(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Huge model (ViT-H/14) CLIP image tower @ 378x378 w/ QuickGELU act
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        pre_norm=True, norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer('vit_huge_patch14_clip_quickgelu_378',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
