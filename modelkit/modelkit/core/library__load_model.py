def _load_model(self, model_name, model_settings=None):
    """
        This function loads dependent models for the current models, populating
        the _models dictionary with the instantiated model objects.
        """
    logger.debug('Loading model', model_name=model_name)
    if model_name in self.models:
        logger.debug('Model already loaded', model_name=model_name)
        return
    configuration = self.configuration[model_name]
    model_dependencies = {}
    for dep_ref_name, dep_name in configuration.model_dependencies.items():
        if dep_name not in self.models:
            self._load_model(dep_name)
        model_dependencies[dep_ref_name] = self.models[dep_name]
    model_settings = {**configuration.model_settings, **self.
        required_models.get(model_name, {})}
    logger.debug('Instantiating Model object', model_name=model_name)
    self.models[model_name] = configuration.model_type(asset_path=self.
        assets_info[configuration.asset].path if configuration.asset else
        '', model_dependencies=model_dependencies, service_settings=self.
        settings, model_settings=model_settings or {}, configuration_key=
        model_name, cache=self.cache)
    logger.debug('Done loading Model', model_name=model_name)
