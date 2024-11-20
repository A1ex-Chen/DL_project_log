@register_model
def vit_giant_patch14_dinov2(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ ViT-G/14 for DINOv2
    """
    model_args = dict(patch_size=14, embed_dim=1536, depth=40, num_heads=24,
        init_values=1e-05, mlp_ratio=2.66667 * 2, mlp_layer=SwiGLUPacked,
        img_size=518, act_layer=nn.SiLU)
    model = _create_vision_transformer('vit_giant_patch14_dinov2',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
