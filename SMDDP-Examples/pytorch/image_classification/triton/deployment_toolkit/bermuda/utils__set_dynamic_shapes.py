def _set_dynamic_shapes(t, shapes):
    for k, v in t.items():
        shape = list(v.shape)
        for dim, s in enumerate(shape):
            if shapes[k][dim] != -1 and shapes[k][dim] != s:
                shapes[k][dim] = -1
