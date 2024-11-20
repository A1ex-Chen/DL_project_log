def _configurations_from_objects(m) ->Dict[str, ModelConfiguration]:
    if inspect.isclass(m) and issubclass(m, Asset):
        return {key: ModelConfiguration(**config, model_type=m) for key,
            config in m.CONFIGURATIONS.items()}
    elif isinstance(m, (list, tuple)):
        return dict(ChainMap(*(_configurations_from_objects(sub_m) for
            sub_m in m)))
    elif isinstance(m, ModuleType):
        return dict(ChainMap(*(_configurations_from_objects(sub_m) for
            sub_m in walk_objects(m))))
    elif isinstance(m, str):
        models = [importlib.import_module(modname) for modname in m.split(',')]
        return _configurations_from_objects(models)
    else:
        raise ValueError(f"Don't know how to configure {m}")
