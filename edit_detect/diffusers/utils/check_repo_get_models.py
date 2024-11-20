def get_models(module, include_pretrained=False):
    """Get the objects in module that are models."""
    models = []
    model_classes = (diffusers.ModelMixin, diffusers.TFModelMixin,
        diffusers.FlaxModelMixin)
    for attr_name in dir(module):
        if not include_pretrained and ('Pretrained' in attr_name or 
            'PreTrained' in attr_name):
            continue
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, model_classes
            ) and attr.__module__ == module.__name__:
            models.append((attr_name, attr))
    return models
