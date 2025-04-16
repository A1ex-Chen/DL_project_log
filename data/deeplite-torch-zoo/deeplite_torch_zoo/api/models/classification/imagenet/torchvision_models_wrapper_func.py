@MODEL_WRAPPER_REGISTRY.register(model_name=register_model_name_key,
    dataset_name='imagenet', task_type='classification')
def wrapper_func(pretrained=False, num_classes=NUM_IMAGENET_CLASSES, **
    model_kwargs):
    LOGGER.warning(
        f'Extra options {model_kwargs} are not applicable for a torchvision model'
        )
    model = torchvision.models.__dict__[model_name_key](pretrained=
        pretrained, num_classes=NUM_IMAGENET_CLASSES)
    if num_classes != NUM_IMAGENET_CLASSES:
        pretrained_dict = model.state_dict()
        model = torchvision.models.__dict__[model_name_key](pretrained=
            False, num_classes=num_classes)
        load_state_dict_partial(model, pretrained_dict)
    return model
