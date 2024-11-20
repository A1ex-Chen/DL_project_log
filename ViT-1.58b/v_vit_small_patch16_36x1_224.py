@register_model
def vit_small_patch16_36x1_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Base w/ LayerScale + 36 x 1 (36 block serial) config. Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    Paper focuses on 24x2 + 48x1 for 'Small' width but those are extremely slow.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=36, num_heads=6,
        init_values=1e-05)
    model = _create_vision_transformer('vit_small_patch16_36x1_224',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
