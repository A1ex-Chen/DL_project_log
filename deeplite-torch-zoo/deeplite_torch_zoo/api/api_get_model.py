def get_model(model_name, dataset_name, pretrained=True, num_classes=None,
    **model_kwargs):
    """
    Tries to find a matching model creation wrapper function in the registry and uses it to create a new model object
    :param model_name: Name of the model to create
    :param dataset_name: Name of dataset the model was trained / is to be trained on
    :param pretrained: Whether to load pretrained weights

    returns a corresponding model object (optionally with pretrained weights)
    """
    model_func = MODEL_WRAPPER_REGISTRY.get(model_name=model_name.lower(),
        dataset_name=dataset_name)
    model_wrapper_kwargs = {'pretrained': pretrained, **model_kwargs}
    if num_classes is not None:
        LOGGER.warning(
            f'Overriding the default number of classes for the model {model_name} on dataset {dataset_name} with num_classes={num_classes}'
            )
        model_wrapper_kwargs.update({'num_classes': num_classes})
    model = model_func(**model_wrapper_kwargs)
    return model
