@register_model
def vit_base_patch16_rpn_224(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-Base (ViT-B/16) w/ residual post-norm
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
        qkv_bias=False, init_values=1e-05, class_token=False, block_fn=
        ResPostBlock, global_pool='avg')
    model = _create_vision_transformer('vit_base_patch16_rpn_224',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
