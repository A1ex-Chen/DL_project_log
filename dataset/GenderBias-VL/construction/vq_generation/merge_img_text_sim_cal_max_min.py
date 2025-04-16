def cal_max_min(data):
    max_val, min_val = 0, 1
    for row in data:
        _sim = float(row['similarity'])
        if _sim > max_val:
            max_val = _sim
        if _sim < min_val and _sim > 0:
            min_val = _sim
    return max_val, min_val
