def _longest_common_prefix(names: List[str]) ->str:
    """
    ["abc.zfg", "abc.zef"] -> "abc."
    """
    names = [n.split('.') for n in names]
    m1, m2 = min(names), max(names)
    ret = [a for a, b in zip(m1, m2) if a == b]
    ret = '.'.join(ret) + '.' if len(ret) else ''
    return ret
