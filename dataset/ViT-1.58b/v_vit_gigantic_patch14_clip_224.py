@register_model
def vit_gigantic_patch14_clip_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-bigG model (ViT-G/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    Pretrained weights from CLIP image tower.
    """
    model_args = dict(patch_size=14, embed_dim=1664, mlp_ratio=64 / 13,
        depth=48, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer('vit_gigantic_patch14_clip_224',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
