def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [(d * (x - 1) + 1) for
            x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p
