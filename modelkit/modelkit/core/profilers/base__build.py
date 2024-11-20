def _build(self, model: typing.Union[Model, AsyncModel, WrappedAsyncModel]):
    """setattr 'profiler' to all sub-models via "model_dependencies" recursively"""
    model.profiler = self
    if isinstance(model, WrappedAsyncModel):
        model = model.async_model
    for model_dependency in model.model_dependencies.values():
        self._build(model_dependency)
