def _longest_common_prefix_str(names: List[str]) ->str:
    m1, m2 = min(names), max(names)
    lcp = [a for a, b in zip(m1, m2) if a == b]
    lcp = ''.join(lcp)
    return lcp
