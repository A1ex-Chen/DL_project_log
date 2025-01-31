def _cfg(url='', **kwargs):
    return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224),
        'pool_size': None, 'crop_pct': 0.9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head', **kwargs}
