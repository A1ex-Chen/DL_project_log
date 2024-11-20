def normalize_depth(depth, d_min, d_max):
    data = np.clip(depth, d_min, d_max)
    return (data - d_min) / (d_max - d_min)
