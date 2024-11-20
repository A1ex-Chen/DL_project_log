@register_model
def flexivit_small(pretrained: bool=False, **kwargs) ->VisionTransformer:
    """ FlexiViT-Small
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6,
        no_embed_class=True)
    model = _create_vision_transformer('flexivit_small', pretrained=
        pretrained, **dict(model_args, **kwargs))
    return model
