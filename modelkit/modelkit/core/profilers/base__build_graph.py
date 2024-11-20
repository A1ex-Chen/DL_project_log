def _build_graph(self, model: typing.Union[Model, AsyncModel,
    WrappedAsyncModel], graph: Dict[str, Set]) ->Dict[str, Set]:
    """Build the model dependency graph in order to compute net cost of all
        sub models. graph[model_name] gives the set of all (direct) sub model names.
        """
    if isinstance(model, WrappedAsyncModel):
        model = model.async_model
    name = model.configuration_key
    children = set()
    for key in model.model_dependencies:
        children.add(key)
        graph = self._build_graph(model.model_dependencies[key], graph)
    graph[name] = children
    return graph
