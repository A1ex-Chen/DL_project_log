def make_wrapper_func(wrapper_fn_name, model_name_key):

    @MODEL_WRAPPER_REGISTRY.register(model_name=f'{model_name_key}_timm',
        dataset_name='imagenet', task_type='classification')
    def wrapper_func(pretrained=False, num_classes=NUM_IMAGENET_CLASSES, **
        model_kwargs):
        model = timm.create_model(model_name_key, pretrained=pretrained,
            num_classes=num_classes, **model_kwargs)
        return model
    wrapper_func.__name__ = wrapper_fn_name
    return wrapper_func
