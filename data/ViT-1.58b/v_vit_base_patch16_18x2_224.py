@register_model
def vit_base_patch16_18x2_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Base w/ LayerScale + 18 x 2 (36 block parallel) config. Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=18, num_heads=12,
        init_values=1e-05, block_fn=ParallelThingsBlock)
    model = _create_vision_transformer('vit_base_patch16_18x2_224',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
