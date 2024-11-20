def list_assets(models: Optional[LibraryModelsType]=None, required_models:
    Optional[List[str]]=None, configuration: Optional[Mapping[str, Union[
    Dict[str, Any], ModelConfiguration]]]=None) ->Set[str]:
    merged_configuration = configure(models=models, configuration=configuration
        )
    assets: Set[str] = set()
    for model in (required_models or merged_configuration.keys()):
        model_configuration = merged_configuration[model]
        if model_configuration.asset:
            assets.add(model_configuration.asset)
        if model_configuration.model_dependencies:
            assets |= list_assets(configuration=merged_configuration,
                required_models=list(model_configuration.model_dependencies
                .values()))
    return assets
