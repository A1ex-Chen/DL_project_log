@register_model
def flexivit_large(pretrained: bool=False, **kwargs) ->VisionTransformer:
    """ FlexiViT-Large
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        no_embed_class=True)
    model = _create_vision_transformer('flexivit_large', pretrained=
        pretrained, **dict(model_args, **kwargs))
    return model
