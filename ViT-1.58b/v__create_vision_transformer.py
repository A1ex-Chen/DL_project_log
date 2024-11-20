def _create_vision_transformer(variant: str, pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    if kwargs.get('features_only', None):
        raise RuntimeError(
            'features_only not implemented for Vision Transformer models.')
    if 'flexi' in variant:
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear',
            antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn
    strict = True
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False
    return build_model_with_cfg(VisionTransformer, variant, pretrained,
        pretrained_filter_fn=_filter_fn, pretrained_strict=strict, **kwargs)
