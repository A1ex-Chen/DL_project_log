def __init__(self, required_models: Optional[List[str]]=None, settings:
    Optional[Union[Dict, LibrarySettings]]=None, assetsmanager_settings:
    Optional[dict]=None, configuration: Optional[Dict[str, Union[Dict[str,
    Any], ModelConfiguration]]]=None, models: Optional[LibraryModelsType]=
    None, route_paths: Optional[Dict[str, str]]=None, **kwargs) ->None:
    super().__init__(required_models=required_models, settings=settings,
        assetsmanager_settings=assetsmanager_settings, configuration=
        configuration, models=models, **kwargs)
    route_paths = route_paths or {}
    for model_name in self.lib.required_models:
        m: AbstractModel = self.lib.get(model_name)
        if not isinstance(m, AbstractModel):
            continue
        path = route_paths.get(model_name, '/predict/' + model_name)
        batch_path = route_paths.get(model_name, '/predict/batch/' + model_name
            )
        summary = ''
        description = ''
        if m.__doc__:
            doclines = m.__doc__.strip().split('\n')
            summary = doclines[0]
            if len(doclines) > 1:
                description = ''.join(doclines[1:])
        console = Console(no_color=True)
        with console.capture() as capture:
            t = m.describe()
            console.print(t)
        description += '\n\n```' + str(capture.get()) + '```'
        logger.info('Adding model', name=model_name)
        item_type = m._item_type or Any
        try:
            item_type.model_json_schema()
        except (ValueError, AttributeError):
            logger.info('Discarding item type info for model', name=
                model_name, path=path)
            item_type = Any
        self.add_api_route(path, self._make_model_endpoint_fn(m, item_type),
            methods=['POST'], description=description, summary=summary,
            tags=[str(type(m).__module__)])
        self.add_api_route(batch_path, self._make_batch_model_endpoint_fn(m,
            item_type), methods=['POST'], description=description, summary=
            summary, tags=[str(type(m).__module__)])
        logger.info('Added model to service', name=model_name, path=path)
