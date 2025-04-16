@MODEL_WRAPPER_REGISTRY.register(model_name=register_model_name_key,
    dataset_name='imagenet', task_type='classification', has_checkpoint=
    register_model_name_key in PYTORCHCV_HAS_CHECKPOINT)
def wrapper_func(pretrained=False, num_classes=NUM_IMAGENET_CLASSES, **
    model_kwargs):
    LOGGER.warning(
        f'Extra options {model_kwargs} are not applicable for a PyTorchCV model'
        )
    model = ptcv_get_model(model_name_key, pretrained=pretrained)
    if num_classes != NUM_IMAGENET_CLASSES:
        pretrained_dict = model.state_dict()
        model = ptcv_get_model(model_name_key, num_classes=num_classes)
        load_state_dict_partial(model, pretrained_dict)
    return model
