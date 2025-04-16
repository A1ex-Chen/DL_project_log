def _create_model(model_kwargs, variant, pretrained=False):
    as_sequential = model_kwargs.pop('as_sequential', False)
    model = MobileNetV3(**model_kwargs)
    if pretrained and model_urls[variant]:
        load_pretrained(model, model_urls[variant])
    if as_sequential:
        model = model.as_sequential()
    return model
