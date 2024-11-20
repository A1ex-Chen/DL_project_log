def expmap_so3(rotvec):
    """Exponential map at identity.
    Create a rotation from canonical coordinates using Rodrigues' formula.
    cfo, 2015/08/13

    """
    theta = numpy.linalg.norm(rotvec)
    axis = rotvec / theta
    return axis_angle(axis, theta)
