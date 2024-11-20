def reshape(t, conv):
    if conv:
        if len(t.shape) == 4:
            return t
        else:
            return t.view(t.shape[0], -1, 1, 1)
    elif len(t.shape) == 4:
        return t.view(t.shape[0], t.shape[1])
    else:
        return t
