@register_model
def vit_huge_patch14_224(pretrained: bool=False, **kwargs) ->VisionTransformer:
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16)
    model = _create_vision_transformer('vit_huge_patch14_224', pretrained=
        pretrained, **dict(model_args, **kwargs))
    return model
