def S_inv_eulerZYX_body_deriv(euler_coordinates, omega):
    """ Compute dE(euler_coordinates)*omega/deuler_coordinates
        cfo, 2015/08/13 

    """
    y = euler_coordinates[1]
    z = euler_coordinates[2]
    """
    w1 = omega[0]; w2 = omega[1]; w3 = omega[2]
    J = numpy.zeros((3,3))
    J[0,0] = 0
    J[0,1] = math.tan(y) / math.cos(y) * (math.sin(z) * w2 + math.cos(z) * w3)
    J[0,2] = w2/math.cos(y)*math.cos(z) - w3/math.cos(y)*math.sin(z)
    J[1,0] = 0
    J[1,1] = 0
    J[1,2] = -w2*math.sin(z) - w3*math.cos(z)
    J[2,0] = w1
    J[2,1] = 1.0/math.cos(y)**2 * (w2 * math.sin(z) + w3 * math.cos(z))
    J[2,2] = w2*math.tan(y)*math.cos(z) - w3*math.tan(y)*math.sin(z)
   
    """
    J_y = numpy.zeros((3, 3))
    J_z = numpy.zeros((3, 3))
    J_y[0, 1] = math.tan(y) / math.cos(y) * math.sin(z)
    J_y[0, 2] = math.tan(y) / math.cos(y) * math.cos(z)
    J_y[2, 1] = math.sin(z) / math.cos(y) ** 2
    J_y[2, 2] = math.cos(z) / math.cos(y) ** 2
    J_z[0, 1] = math.cos(z) / math.cos(y)
    J_z[0, 2] = -math.sin(z) / math.cos(y)
    J_z[1, 1] = -math.sin(z)
    J_z[1, 2] = -math.cos(z)
    J_z[2, 1] = math.cos(z) * math.tan(y)
    J_z[2, 2] = -math.sin(z) * math.tan(y)
    J = numpy.zeros((3, 3))
    J[:, 1] = numpy.dot(J_y, omega)
    J[:, 2] = numpy.dot(J_z, omega)
    return J
