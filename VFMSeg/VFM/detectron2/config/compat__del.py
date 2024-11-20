def _del(key_seq: List[str]) ->None:
    cur = cfg
    for k in key_seq[:-1]:
        cur = cur[k]
    del cur[key_seq[-1]]
    if len(cur) == 0 and len(key_seq) > 1:
        _del(key_seq[:-1])
