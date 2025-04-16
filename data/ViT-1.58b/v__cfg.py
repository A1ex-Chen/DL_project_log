def _cfg(url: str='', **kwargs) ->Dict[str, Any]:
    return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224),
        'pool_size': None, 'crop_pct': 0.9, 'interpolation': 'bicubic',
        'fixed_input_size': True, 'mean': IMAGENET_INCEPTION_MEAN, 'std':
        IMAGENET_INCEPTION_STD, 'first_conv': 'patch_embed.proj',
        'classifier': 'head', **kwargs}
