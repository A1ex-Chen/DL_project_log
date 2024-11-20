def _has(name: str) ->bool:
    cur = cfg
    for n in name.split('.'):
        if n not in cur:
            return False
        cur = cur[n]
    return True
