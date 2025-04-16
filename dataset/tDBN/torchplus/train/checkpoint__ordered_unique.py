def _ordered_unique(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]
