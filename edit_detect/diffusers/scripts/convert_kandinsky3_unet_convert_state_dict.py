def convert_state_dict(unet_state_dict):
    """
    Convert the state dict of a U-Net model to match the key format expected by Kandinsky3UNet model.
    Args:
        unet_model (torch.nn.Module): The original U-Net model.
        unet_kandi3_model (torch.nn.Module): The Kandinsky3UNet model to match keys with.

    Returns:
        OrderedDict: The converted state dictionary.
    """
    converted_state_dict = {}
    for key in unet_state_dict:
        new_key = key
        for pattern, new_pattern in MAPPING.items():
            new_key = new_key.replace(pattern, new_pattern)
        for dyn_pattern, dyn_new_pattern in DYNAMIC_MAP.items():
            has_matched = False
            if fnmatch.fnmatch(new_key, f'*.{dyn_pattern}.*'
                ) and not has_matched:
                star = int(new_key.split(dyn_pattern.split('.')[0])[-1].
                    split('.')[1])
                if isinstance(dyn_new_pattern, tuple):
                    new_star = star + dyn_new_pattern[-1]
                    dyn_new_pattern = dyn_new_pattern[0]
                else:
                    new_star = star
                pattern = dyn_pattern.replace('*', str(star))
                new_pattern = dyn_new_pattern.replace('*', str(new_star))
                new_key = new_key.replace(pattern, new_pattern)
                has_matched = True
        converted_state_dict[new_key] = unet_state_dict[key]
    return converted_state_dict
