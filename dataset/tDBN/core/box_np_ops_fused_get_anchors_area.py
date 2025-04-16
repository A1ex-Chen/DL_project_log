@numba.jit(nopython=True)
def fused_get_anchors_area(dense_map, anchors_bv, stride, offset, grid_size):
    anchor_coor = np.zeros(anchors_bv.shape[1:], dtype=np.int32)
    grid_size_x = grid_size[0] - 1
    grid_size_y = grid_size[1] - 1
    N = anchors_bv.shape[0]
    ret = np.zeros(N, dtype=dense_map.dtype)
    for i in range(N):
        anchor_coor[0] = np.floor((anchors_bv[i, 0] - offset[0]) / stride[0])
        anchor_coor[1] = np.floor((anchors_bv[i, 1] - offset[1]) / stride[1])
        anchor_coor[2] = np.floor((anchors_bv[i, 2] - offset[0]) / stride[0])
        anchor_coor[3] = np.floor((anchors_bv[i, 3] - offset[1]) / stride[1])
        anchor_coor[0] = max(anchor_coor[0], 0)
        anchor_coor[1] = max(anchor_coor[1], 0)
        anchor_coor[2] = min(anchor_coor[2], grid_size_x)
        anchor_coor[3] = min(anchor_coor[3], grid_size_y)
        ID = dense_map[anchor_coor[3], anchor_coor[2]]
        IA = dense_map[anchor_coor[1], anchor_coor[0]]
        IB = dense_map[anchor_coor[3], anchor_coor[0]]
        IC = dense_map[anchor_coor[1], anchor_coor[2]]
        ret[i] = ID - IB - IC + IA
    return ret
