def reshape_weight_for_sd(w):
    if not w.ndim == 1:
        return w.reshape(*w.shape, 1, 1)
    else:
        return w
