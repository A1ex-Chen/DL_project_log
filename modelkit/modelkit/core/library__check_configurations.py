def _check_configurations(self, configuration_key):
    if configuration_key not in self.configuration:
        logger.error('Cannot find model configuration', name=
            configuration_key, sentry=True)
        candidates = {x: collections.Counter(x) for x in self.configuration}
        configuration = collections.Counter(configuration_key)
        differences = sorted((sum(x for x in (configuration - candidate).
            values()) + sum(x for x in (candidate - configuration).values()
            ), key) for key, candidate in candidates.items())
        msg = f"""Cannot resolve assets for model `{configuration_key}`: configuration not found.

Configured models: {', '.join(sorted(self.configuration))}.

"""
        if differences and differences[0] and differences[0][0] < 10:
            msg += f'Did you mean "{differences[0][1]}" ?\n'
        raise ConfigurationNotFoundException(msg)
    configuration = self.configuration[configuration_key]
    for dep_name in configuration.model_dependencies.values():
        self._check_configurations(dep_name)
