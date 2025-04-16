def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]
    name = obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError(''.join(failed))
    if name in ['VersatileDiffusionTextToImagePipeline',
        'VersatileDiffusionPipeline',
        'VersatileDiffusionDualGuidedPipeline',
        'StableDiffusionImageVariationPipeline', 'UnCLIPPipeline'
        ] and is_transformers_version('<', '4.25.0'):
        raise ImportError(
            f"""You need to install `transformers>=4.25` in order to use {name}: 
```
 pip install --upgrade transformers 
```"""
            )
    if name in ['StableDiffusionDepth2ImgPipeline',
        'StableDiffusionPix2PixZeroPipeline'] and is_transformers_version('<',
        '4.26.0'):
        raise ImportError(
            f"""You need to install `transformers>=4.26` in order to use {name}: 
```
 pip install --upgrade transformers 
```"""
            )
