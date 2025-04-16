def get_same_padding(x: int, k: int, s: int, d: int):
    return max((int(math.ceil(x // s)) - 1) * s + (k - 1) * d + 1 - x, 0)
