@register_model
def vit_base_patch16_224_miil(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
        qkv_bias=False)
    model = _create_vision_transformer('vit_base_patch16_224_miil',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
