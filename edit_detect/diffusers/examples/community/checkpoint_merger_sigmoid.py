@staticmethod
def sigmoid(theta0, theta1, theta2, alpha):
    alpha = alpha * alpha * (3 - 2 * alpha)
    return theta0 + (theta1 - theta0) * alpha
