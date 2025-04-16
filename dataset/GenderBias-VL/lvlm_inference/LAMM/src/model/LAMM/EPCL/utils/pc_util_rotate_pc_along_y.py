def rotate_pc_along_y(pc, rot_angle):
    """Input ps is NxC points with first 3 channels as XYZ
    z is facing forward, x is left ward, y is downward
    """
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc
