def right_jacobian_so3(rotvec):
    """Right Jacobian for Exponential map in SO(3)
    Equation (10.86) and following equations in G.S. Chirikjian, "Stochastic
    Models, Information Theory, and Lie Groups", Volume 2, 2008.

    > expmap_so3(thetahat + omega) pprox expmap_so3(thetahat) * expmap_so3(Jr * omega)
    where Jr = right_jacobian_so3(thetahat);
    This maps a perturbation in the tangent space (omega) to a perturbation
    on the manifold (expmap_so3(Jr * omega))
    cfo, 2015/08/13

    """
    theta2 = numpy.dot(rotvec, rotvec)
    if theta2 <= _EPS:
        return numpy.identity(3, dtype=numpy.float64)
    else:
        theta = numpy.sqrt(theta2)
        Y = skew(rotvec) / theta
        I_3x3 = numpy.identity(3, dtype=numpy.float64)
        J_r = I_3x3 - (1.0 - numpy.cos(theta)) / theta * Y + (1.0 - numpy.
            sin(theta) / theta) * numpy.dot(Y, Y)
        return J_r
