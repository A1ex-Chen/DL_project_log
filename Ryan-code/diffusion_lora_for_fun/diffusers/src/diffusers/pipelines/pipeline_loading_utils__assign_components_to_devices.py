def _assign_components_to_devices(module_sizes: Dict[str, float],
    device_memory: Dict[str, float], device_mapping_strategy: str='balanced'):
    device_ids = list(device_memory.keys())
    device_cycle = device_ids + device_ids[::-1]
    device_memory = device_memory.copy()
    device_id_component_mapping = {}
    current_device_index = 0
    for component in module_sizes:
        device_id = device_cycle[current_device_index % len(device_cycle)]
        component_memory = module_sizes[component]
        curr_device_memory = device_memory[device_id]
        if component_memory > curr_device_memory:
            device_id_component_mapping['cpu'] = [component]
        else:
            if device_id not in device_id_component_mapping:
                device_id_component_mapping[device_id] = [component]
            else:
                device_id_component_mapping[device_id].append(component)
            device_memory[device_id] -= component_memory
            current_device_index += 1
    return device_id_component_mapping
