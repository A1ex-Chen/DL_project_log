def get_3d_box_batch(box_size, heading_angle, center):
    """box_size: [x1,x2,...,xn,3] -- box dimensions without flipping [X, Y, Z] -- l, w, h
        heading_angle: [x1,x2,...,xn] -- theta in radians
        center: [x1,x2,...,xn,3] -- center point has been flipped to camera axis [X, -Z, Y]
    Return:
        [x1,x3,...,xn,8,3]
    """
    input_shape = heading_angle.shape
    R = roty_batch(heading_angle)
    l = np.expand_dims(box_size[..., 0], -1)
    w = np.expand_dims(box_size[..., 1], -1)
    h = np.expand_dims(box_size[..., 2], -1)
    corners_3d = np.zeros(tuple(list(input_shape) + [8, 3]))
    corners_3d[..., :, 0] = np.concatenate((l / 2, l / 2, -l / 2, -l / 2, l /
        2, l / 2, -l / 2, -l / 2), -1)
    corners_3d[..., :, 1] = np.concatenate((h / 2, h / 2, h / 2, h / 2, -h /
        2, -h / 2, -h / 2, -h / 2), -1)
    corners_3d[..., :, 2] = np.concatenate((w / 2, -w / 2, -w / 2, w / 2, w /
        2, -w / 2, -w / 2, w / 2), -1)
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape) + 1, len(input_shape)]
    corners_3d = np.matmul(corners_3d, np.transpose(R, tuple(tlist)))
    corners_3d += np.expand_dims(center, -2)
    return corners_3d
