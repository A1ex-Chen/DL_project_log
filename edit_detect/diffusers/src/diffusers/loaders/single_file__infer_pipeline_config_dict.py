def _infer_pipeline_config_dict(pipeline_class):
    parameters = inspect.signature(pipeline_class.__init__).parameters
    required_parameters = {k: v for k, v in parameters.items() if v.default ==
        inspect._empty}
    component_types = pipeline_class._get_signature_types()
    component_types = {k: v for k, v in component_types.items() if k in
        required_parameters}
    config_dict = _map_component_types_to_config_dict(component_types)
    return config_dict
