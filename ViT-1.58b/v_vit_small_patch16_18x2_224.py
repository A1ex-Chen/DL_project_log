@register_model
def vit_small_patch16_18x2_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Small w/ LayerScale + 18 x 2 (36 block parallel) config. Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    Paper focuses on 24x2 + 48x1 for 'Small' width but those are extremely slow.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=18, num_heads=6,
        init_values=1e-05, block_fn=ParallelThingsBlock)
    model = _create_vision_transformer('vit_small_patch16_18x2_224',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
