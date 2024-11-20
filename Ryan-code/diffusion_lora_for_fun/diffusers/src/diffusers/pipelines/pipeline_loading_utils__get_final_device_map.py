def _get_final_device_map(device_map, pipeline_class, passed_class_obj,
    init_dict, library, max_memory, **kwargs):
    from diffusers import pipelines
    torch_dtype = kwargs.get('torch_dtype', torch.float32)
    init_empty_modules = {}
    for name, (library_name, class_name) in init_dict.items():
        if class_name.startswith('Flax'):
            raise ValueError(
                'Flax pipelines are not supported with `device_map`.')
        is_pipeline_module = hasattr(pipelines, library_name)
        importable_classes = ALL_IMPORTABLE_CLASSES
        loaded_sub_model = None
        if name in passed_class_obj:
            maybe_raise_or_warn(library_name, library, class_name,
                importable_classes, passed_class_obj, name, is_pipeline_module)
            with accelerate.init_empty_weights():
                loaded_sub_model = passed_class_obj[name]
        else:
            loaded_sub_model = _load_empty_model(library_name=library_name,
                class_name=class_name, importable_classes=
                importable_classes, pipelines=pipelines, is_pipeline_module
                =is_pipeline_module, pipeline_class=pipeline_class, name=
                name, torch_dtype=torch_dtype, cached_folder=kwargs.get(
                'cached_folder', None), force_download=kwargs.get(
                'force_download', None), resume_download=kwargs.get(
                'resume_download', None), proxies=kwargs.get('proxies',
                None), local_files_only=kwargs.get('local_files_only', None
                ), token=kwargs.get('token', None), revision=kwargs.get(
                'revision', None))
        if loaded_sub_model is not None:
            init_empty_modules[name] = loaded_sub_model
    module_sizes = {module_name: compute_module_sizes(module, dtype=
        torch_dtype)[''] for module_name, module in init_empty_modules.
        items() if isinstance(module, torch.nn.Module)}
    module_sizes = dict(sorted(module_sizes.items(), key=lambda item: item[
        1], reverse=True))
    max_memory = get_max_memory(max_memory)
    max_memory = dict(sorted(max_memory.items(), key=lambda item: item[1],
        reverse=True))
    max_memory = {k: v for k, v in max_memory.items() if k != 'cpu'}
    final_device_map = None
    if len(max_memory) > 0:
        device_id_component_mapping = _assign_components_to_devices(
            module_sizes, max_memory, device_mapping_strategy=device_map)
        final_device_map = {}
        for device_id, components in device_id_component_mapping.items():
            for component in components:
                final_device_map[component] = device_id
    return final_device_map
