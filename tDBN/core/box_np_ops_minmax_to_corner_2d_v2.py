def minmax_to_corner_2d_v2(minmax_box):
    return minmax_box[..., [0, 1, 0, 3, 2, 3, 2, 1]].reshape(-1, 4, 2)
