def flip_axis_to_camera_np(pc):
    """Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    """
    pc2 = pc.copy()
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]
    pc2[..., 1] *= -1
    return pc2
