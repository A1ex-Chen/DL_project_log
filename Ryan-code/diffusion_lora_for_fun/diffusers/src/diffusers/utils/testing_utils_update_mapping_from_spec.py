def update_mapping_from_spec(device_fn_dict: Dict[str, Callable],
    attribute_name: str):
    try:
        spec_fn = getattr(device_spec_module, attribute_name)
        device_fn_dict[torch_device] = spec_fn
    except AttributeError as e:
        if 'default' not in device_fn_dict:
            raise AttributeError(
                f"`{attribute_name}` not found in '{device_spec_path}' and no default fallback function found."
                ) from e
