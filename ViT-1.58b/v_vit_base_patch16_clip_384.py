@register_model
def vit_base_patch16_clip_384(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-B/16 CLIP image tower @ 384x384
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
        pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer('vit_base_patch16_clip_384',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
