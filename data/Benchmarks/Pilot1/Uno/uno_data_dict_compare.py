def dict_compare(d1, d2, ignore=[], expand=False):
    d1_keys = set(d1.keys()) - set(ignore)
    d2_keys = set(d2.keys()) - set(ignore)
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = set({x: (d1[x], d2[x]) for x in intersect_keys if d1[x] !=
        d2[x]})
    common = set(x for x in intersect_keys if d1[x] == d2[x])
    equal = not (added or removed or modified)
    if expand:
        return equal, added, removed, modified, common
    else:
        return equal, added | removed | modified
