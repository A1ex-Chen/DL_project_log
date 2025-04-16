def patch(model, target, updater, *args, **kwargs):
    """
    recursively (post-order) update all modules with the target type and its
    subclasses, make a initialization/composition/inheritance/... via the
    updater.create_from.
    """
    for name, module in model.named_children():
        model._modules[name] = patch(module, target, updater, *args, **kwargs)
    if isinstance(model, target):
        return updater.create_from(model, *args, **kwargs)
    return model
