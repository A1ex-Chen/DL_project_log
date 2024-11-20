def _expand_param_groups(params: List[Dict[str, Any]]) ->List[Dict[str, Any]]:
    ret = defaultdict(dict)
    for item in params:
        assert 'params' in item
        cur_params = {x: y for x, y in item.items() if x != 'params'}
        for param in item['params']:
            ret[param].update({'params': [param], **cur_params})
    return list(ret.values())
