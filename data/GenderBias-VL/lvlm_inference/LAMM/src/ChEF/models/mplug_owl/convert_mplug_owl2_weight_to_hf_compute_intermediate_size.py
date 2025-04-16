def compute_intermediate_size(n):
    return int(math.ceil(n * 8 / 3) + 255) // 256 * 256
