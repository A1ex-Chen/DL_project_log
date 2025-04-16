@staticmethod
def inv_sigmoid(theta0, theta1, theta2, alpha):
    import math
    alpha = 0.5 - math.sin(math.asin(1.0 - 2.0 * alpha) / 3.0)
    return theta0 + (theta1 - theta0) * alpha
