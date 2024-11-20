def configure(models: Optional[LibraryModelsType]=None, configuration:
    Optional[Mapping[str, Union[Dict[str, Any], ModelConfiguration]]]=None
    ) ->Dict[str, ModelConfiguration]:
    if not models:
        models = os.environ.get('MODELKIT_DEFAULT_PACKAGE')
    conf = _configurations_from_objects(models) if models else {}
    if configuration:
        for key in (set(conf.keys()) & set(configuration.keys())):
            if key in configuration:
                conf_value = configuration[key]
                if isinstance(conf_value, ModelConfiguration):
                    conf[key] = conf_value
                elif isinstance(conf_value, dict):
                    conf[key] = ModelConfiguration(**{**conf[key].
                        model_dump(), **conf_value})
        for key in (set(configuration.keys()) - set(conf.keys())):
            conf_value = configuration[key]
            if isinstance(conf_value, ModelConfiguration):
                conf[key] = conf_value
            elif isinstance(conf_value, dict):
                conf[key] = ModelConfiguration(**conf_value)
    return conf
