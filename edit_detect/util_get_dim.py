def get_dim(x, dim, nd):
    if nd != None:
        return list(range(len(x.shape)))[nd:]
    else:
        return dim
