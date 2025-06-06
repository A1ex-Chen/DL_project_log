def make_wrapper_func(wrapper_fn, wrapper_name, model_name, dataset_name=
    'imagenet', num_classes=NUM_IMAGENET_CLASSES, has_checkpoint=True):

    @MODEL_WRAPPER_REGISTRY.register(model_name=model_name, dataset_name=
        dataset_name, task_type='classification', has_checkpoint=has_checkpoint
        )
    def wrapper_func(pretrained=False, num_classes=num_classes):
        return wrapper_fn(model_name=model_name, num_classes=num_classes,
            pretrained=pretrained)
    wrapper_func.__name__ = wrapper_name
    return wrapper_func
