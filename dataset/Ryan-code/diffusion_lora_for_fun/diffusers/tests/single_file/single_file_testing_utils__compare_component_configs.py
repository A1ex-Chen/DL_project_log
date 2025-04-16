def _compare_component_configs(self, pipe, single_file_pipe):
    if pipe.text_encoder:
        for param_name, param_value in single_file_pipe.text_encoder.config.to_dict(
            ).items():
            if param_name in ['torch_dtype', 'architectures', '_name_or_path']:
                continue
            assert pipe.text_encoder.config.to_dict()[param_name
                ] == param_value
    for param_name, param_value in single_file_pipe.text_encoder_2.config.to_dict(
        ).items():
        if param_name in ['torch_dtype', 'architectures', '_name_or_path']:
            continue
        assert pipe.text_encoder_2.config.to_dict()[param_name] == param_value
    PARAMS_TO_IGNORE = ['torch_dtype', '_name_or_path', 'architectures',
        '_use_default_values', '_diffusers_version']
    for component_name, component in single_file_pipe.components.items():
        if component_name in single_file_pipe._optional_components:
            continue
        if component_name in ['text_encoder', 'text_encoder_2', 'tokenizer',
            'tokenizer_2']:
            continue
        if component_name in ['safety_checker', 'feature_extractor']:
            continue
        assert component_name in pipe.components, f'single file {component_name} not found in pretrained pipeline'
        assert isinstance(component, pipe.components[component_name].__class__
            ), f'single file {component.__class__.__name__} and pretrained {pipe.components[component_name].__class__.__name__} are not the same'
        for param_name, param_value in component.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            if param_name == 'upcast_attention' and pipe.components[
                component_name].config[param_name] is None:
                pipe.components[component_name].config[param_name
                    ] = param_value
            assert pipe.components[component_name].config[param_name
                ] == param_value, f'single file {param_name}: {param_value} differs from pretrained {pipe.components[component_name].config[param_name]}'
