def get_position_angle_vec(position):
    return [(position / np.power(10000, 2 * (hid_j // 2) / d_hid)) for
        hid_j in range(d_hid)]
