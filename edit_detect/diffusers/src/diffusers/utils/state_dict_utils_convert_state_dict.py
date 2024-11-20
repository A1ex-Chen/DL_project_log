def convert_state_dict(state_dict, mapping):
    """
    Simply iterates over the state dict and replaces the patterns in `mapping` with the corresponding values.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dict to convert.
        mapping (`dict[str, str]`):
            The mapping to use for conversion, the mapping should be a dictionary with the following structure:
                - key: the pattern to replace
                - value: the pattern to replace with

    Returns:
        converted_state_dict (`dict`)
            The converted state dict.
    """
    converted_state_dict = {}
    for k, v in state_dict.items():
        for pattern in KEYS_TO_ALWAYS_REPLACE.keys():
            if pattern in k:
                new_pattern = KEYS_TO_ALWAYS_REPLACE[pattern]
                k = k.replace(pattern, new_pattern)
        for pattern in mapping.keys():
            if pattern in k:
                new_pattern = mapping[pattern]
                k = k.replace(pattern, new_pattern)
                break
        converted_state_dict[k] = v
    return converted_state_dict
