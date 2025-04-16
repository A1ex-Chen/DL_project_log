@register_model
def vit_huge_patch14_clip_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Huge model (ViT-H/14) CLIP image tower.
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer('vit_huge_patch14_clip_224',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
