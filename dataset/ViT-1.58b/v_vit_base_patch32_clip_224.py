@register_model
def vit_base_patch32_clip_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-B/32 CLIP image tower @ 224x224
    """
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12,
        pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer('vit_base_patch32_clip_224',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
