def heading2rotmat(heading_angle):
    pass
    rotmat = np.zeros((3, 3))
    rotmat[1, 1] = 1
    cosval = np.cos(heading_angle)
    sinval = np.sin(heading_angle)
    rotmat[0, :] = np.array([cosval, 0, sinval])
    rotmat[2, :] = np.array([-sinval, 0, cosval])
    return rotmat
