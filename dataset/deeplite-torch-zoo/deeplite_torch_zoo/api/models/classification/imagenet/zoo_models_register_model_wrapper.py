def register_model_wrapper(model_fn, model_name, **model_init_kwargs):

    def get_model(num_classes=NUM_IMAGENET_CLASSES, pretrained=False, **
        model_kwargs):
        LOGGER.warning(
            f'Extra options {model_kwargs} are not applicable for a deeplite-torch-zoo model'
            )
        model = model_fn(num_classes=num_classes, **model_init_kwargs)
        if pretrained:
            model = load_checkpoint(model, model_name, 'imagenet',
                CHECKPOINT_URLS)
        return model
    get_model = MODEL_WRAPPER_REGISTRY.register(model_name=model_name,
        dataset_name='imagenet', task_type='classification', has_checkpoint
        =f'{model_name}_imagenet' in CHECKPOINT_URLS)(get_model)
    get_model.__name__ = f'{model_name}_imagenet'
    return get_model
