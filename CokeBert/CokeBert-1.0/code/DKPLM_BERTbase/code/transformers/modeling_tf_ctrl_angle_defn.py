def angle_defn(pos, i, d_model_size):
    angle_rates = 1 / np.power(10000, 2 * (i // 2) / np.float32(d_model_size))
    return pos * angle_rates
