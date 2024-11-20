def get(self, name, model_type: Optional[Type[T]]=None) ->T:
    """
        Get a model by name

        :param name: The name of the required model
        :return: required model
        """
    if self._lazy_loading:
        if name not in self.models:
            self._load(name)
        if not self.models[name]._loaded:
            self.models[name].load()
    if name not in self.models:
        raise errors.ModelsNotFound(f'Model `{name}` not loaded.' + (
            f" (loaded models: {', '.join(self.models)})." if self.models else
            '.'))
    m = self.models[name]
    if model_type and not isinstance(m, model_type):
        raise ValueError(f'Model `{m}` is not an instance of {model_type}')
    return cast(T, m)
