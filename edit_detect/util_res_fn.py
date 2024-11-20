def res_fn(x):
    if nd != None:
        used_dim = list(range(len(x.shape)))[nd:]
        usded_out_dimameter = max(list(x.shape[nd:])) * thres
    elif dim != None:
        used_dim = dim
        usded_out_dimameter = max([x.shape[d] for d in dim]) * thres
    return partial(filtered_by_freq, dim=used_dim, in_diamiter=0,
        out_diamiter=usded_out_dimameter)
