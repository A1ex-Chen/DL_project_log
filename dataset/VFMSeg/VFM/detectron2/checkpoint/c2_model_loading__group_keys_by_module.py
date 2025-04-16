def _group_keys_by_module(keys: List[str], original_names: Dict[str, str]):
    """
    Params in the same submodule are grouped together.

    Args:
        keys: names of all parameters
        original_names: mapping from parameter name to their name in the checkpoint

    Returns:
        dict[name -> all other names in the same group]
    """

    def _submodule_name(key):
        pos = key.rfind('.')
        if pos < 0:
            return None
        prefix = key[:pos + 1]
        return prefix
    all_submodules = [_submodule_name(k) for k in keys]
    all_submodules = [x for x in all_submodules if x]
    all_submodules = sorted(all_submodules, key=len)
    ret = {}
    for prefix in all_submodules:
        group = [k for k in keys if k.startswith(prefix)]
        if len(group) <= 1:
            continue
        original_name_lcp = _longest_common_prefix_str([original_names[k] for
            k in group])
        if len(original_name_lcp) == 0:
            continue
        for k in group:
            if k in ret:
                continue
            ret[k] = group
    return ret
