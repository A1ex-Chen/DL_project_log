def intersect_dicts(da, db, exclude=()):
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in
        exclude) and v.shape == db[k].shape}
