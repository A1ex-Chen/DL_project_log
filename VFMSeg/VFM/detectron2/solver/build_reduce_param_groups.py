def reduce_param_groups(params: List[Dict[str, Any]]) ->List[Dict[str, Any]]:
    params = _expand_param_groups(params)
    groups = defaultdict(list)
    for item in params:
        cur_params = tuple((x, y) for x, y in item.items() if x != 'params')
        groups[cur_params].extend(item['params'])
    ret = []
    for param_keys, param_values in groups.items():
        cur = {kv[0]: kv[1] for kv in param_keys}
        cur['params'] = param_values
        ret.append(cur)
    return ret
