@register_model
def flexivit_base(pretrained: bool=False, **kwargs) ->VisionTransformer:
    """ FlexiViT-Base
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
        no_embed_class=True)
    model = _create_vision_transformer('flexivit_base', pretrained=
        pretrained, **dict(model_args, **kwargs))
    return model
