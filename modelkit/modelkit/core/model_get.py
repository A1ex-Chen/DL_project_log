def get(self, key: str, model_type: Optional[Type[ModelDependency]]=None
    ) ->ModelDependency:
    m = self.models[key]
    if model_type and not isinstance(m, model_type):
        raise ValueError(f'Model `{m}` is not an instance of {model_type}')
    return cast(ModelDependency, m)
