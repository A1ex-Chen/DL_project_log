def get_posi_angle_vec(position):
    return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
