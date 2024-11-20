def _map_component_types_to_config_dict(component_types):
    diffusers_module = importlib.import_module(__name__.split('.')[0])
    config_dict = {}
    component_types.pop('self', None)
    if is_transformers_available():
        transformers_version = version.parse(version.parse(transformers.
            __version__).base_version)
    else:
        transformers_version = 'N/A'
    for component_name, component_value in component_types.items():
        is_diffusers_model = issubclass(component_value[0],
            diffusers_module.ModelMixin)
        is_scheduler_enum = component_value[0
            ].__name__ == 'KarrasDiffusionSchedulers'
        is_scheduler = issubclass(component_value[0], diffusers_module.
            SchedulerMixin)
        is_transformers_model = is_transformers_available() and issubclass(
            component_value[0], PreTrainedModel
            ) and transformers_version >= version.parse('4.20.0')
        is_transformers_tokenizer = is_transformers_available() and issubclass(
            component_value[0], PreTrainedTokenizer
            ) and transformers_version >= version.parse('4.20.0')
        if (is_diffusers_model and component_name not in
            SINGLE_FILE_OPTIONAL_COMPONENTS):
            config_dict[component_name] = ['diffusers', component_value[0].
                __name__]
        elif is_scheduler_enum or is_scheduler:
            if is_scheduler_enum:
                config_dict[component_name] = ['diffusers', 'DDIMScheduler']
            elif is_scheduler:
                config_dict[component_name] = ['diffusers', component_value
                    [0].__name__]
        elif (is_transformers_model or is_transformers_tokenizer
            ) and component_name not in SINGLE_FILE_OPTIONAL_COMPONENTS:
            config_dict[component_name] = ['transformers', component_value[
                0].__name__]
        else:
            config_dict[component_name] = [None, None]
    return config_dict
