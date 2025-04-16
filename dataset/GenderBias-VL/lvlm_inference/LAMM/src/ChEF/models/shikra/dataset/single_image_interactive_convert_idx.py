def convert_idx(objs_seq, objs_value, get_obj_idx_func):
    if objs_seq is None:
        return None
    ret = []
    for objs_idx in objs_seq:
        new_objs_idx = []
        for idx in objs_idx:
            new_idx = get_obj_idx_func(objs_value[idx])
            new_objs_idx.append(new_idx)
        ret.append(tuple(new_objs_idx))
    return tuple(ret)
