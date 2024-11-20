def clip(a: float):
    if a > 1:
        a = 1
    if a < 0:
        a = 0
    return a
