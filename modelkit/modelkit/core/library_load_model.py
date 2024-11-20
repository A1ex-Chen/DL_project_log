def load_model(model_name, configuration: Optional[Dict[str, Union[Dict[str,
    Any], ModelConfiguration]]]=None, models: Optional[LibraryModelsType]=
    None, model_type: Optional[Type[T]]=None) ->T:
    """
    Loads an modelkit model without the need for a ModelLibrary.
    This is useful for development, and should be avoided in production
    code.
    """
    lib = ModelLibrary(required_models=[model_name], models=models,
        configuration=configuration, settings={'lazy_loading': True})
    return lib.get(model_name, model_type=model_type)
