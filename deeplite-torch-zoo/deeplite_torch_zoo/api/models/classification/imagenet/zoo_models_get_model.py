def get_model(num_classes=NUM_IMAGENET_CLASSES, pretrained=False, **
    model_kwargs):
    LOGGER.warning(
        f'Extra options {model_kwargs} are not applicable for a deeplite-torch-zoo model'
        )
    model = model_fn(num_classes=num_classes, **model_init_kwargs)
    if pretrained:
        model = load_checkpoint(model, model_name, 'imagenet', CHECKPOINT_URLS)
    return model
