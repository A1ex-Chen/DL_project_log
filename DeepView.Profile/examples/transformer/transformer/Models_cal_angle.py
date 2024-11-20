def cal_angle(position, hid_idx):
    return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
