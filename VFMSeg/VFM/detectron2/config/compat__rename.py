def _rename(cfg: CN, old: str, new: str) ->None:
    old_keys = old.split('.')
    new_keys = new.split('.')

    def _set(key_seq: List[str], val: str) ->None:
        cur = cfg
        for k in key_seq[:-1]:
            if k not in cur:
                cur[k] = CN()
            cur = cur[k]
        cur[key_seq[-1]] = val

    def _get(key_seq: List[str]) ->CN:
        cur = cfg
        for k in key_seq:
            cur = cur[k]
        return cur

    def _del(key_seq: List[str]) ->None:
        cur = cfg
        for k in key_seq[:-1]:
            cur = cur[k]
        del cur[key_seq[-1]]
        if len(cur) == 0 and len(key_seq) > 1:
            _del(key_seq[:-1])
    _set(new_keys, _get(old_keys))
    _del(old_keys)
