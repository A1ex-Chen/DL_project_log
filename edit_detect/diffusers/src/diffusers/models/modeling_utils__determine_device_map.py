def _determine_device_map(model: 'ModelMixin', device_map, max_memory,
    torch_dtype):
    if isinstance(device_map, str):
        no_split_modules = model._get_no_split_modules(device_map)
        device_map_kwargs = {'no_split_module_classes': no_split_modules}
        if device_map != 'sequential':
            max_memory = get_balanced_memory(model, dtype=torch_dtype,
                low_zero=device_map == 'balanced_low_0', max_memory=
                max_memory, **device_map_kwargs)
        else:
            max_memory = get_max_memory(max_memory)
        device_map_kwargs['max_memory'] = max_memory
        device_map = infer_auto_device_map(model, dtype=torch_dtype, **
            device_map_kwargs)
    return device_map
