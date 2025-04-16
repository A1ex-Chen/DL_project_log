def extract_pc_in_box3d(pc, box3d):
    """pc: (N,3), box3d: (8,3)"""
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds
