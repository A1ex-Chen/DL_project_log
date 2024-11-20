def _get(key_seq: List[str]) ->CN:
    cur = cfg
    for k in key_seq:
        cur = cur[k]
    return cur
