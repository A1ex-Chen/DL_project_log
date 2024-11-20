def _key(v):
    maj_v, min_v = cls._parse_version(v)
    if min_v is None:
        min_v = 0
    return maj_v, min_v
