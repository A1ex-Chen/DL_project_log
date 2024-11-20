def load(model_name, **model_config):
    """
    Load a model by name.

    The function searches subclasses of BaseDetectionModel for a match with the given name. If a match is found, an instance of the model is returned. If no match is found, a warning is logged and the function returns None.

    Args:
        model_name (str): The name of the model to load.
        model_config (dict, optional): The model configuration to use when instantiating the model.

    Returns:
        BaseDetectionModel: An instance of the requested model, or None if no match was found.
    """
    for cls in inheritors(BaseMotionModel):
        if model_name in [cls.__name__.lower(), cls.__name__]:
            filtered_config = {k.lower(): v for k, v in model_config.items(
                ) if k.lower() in inspect.signature(cls.__init__).parameters}
            return cls(**filtered_config)
    logger.warning(
        f'Model {model_name} not found. Available models: {[cls.__name__ for cls in inheritors(BaseMotionModel)]} (lowercase is allowed)'
        )
