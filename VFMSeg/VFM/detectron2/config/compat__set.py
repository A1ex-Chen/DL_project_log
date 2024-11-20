def _set(key_seq: List[str], val: str) ->None:
    cur = cfg
    for k in key_seq[:-1]:
        if k not in cur:
            cur[k] = CN()
        cur = cur[k]
    cur[key_seq[-1]] = val
