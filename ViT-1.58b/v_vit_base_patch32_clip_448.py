@register_model
def vit_base_patch32_clip_448(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-B/32 CLIP image tower @ 448x448
    """
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12,
        pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer('vit_base_patch32_clip_448',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
