def lagrange_polynomial_coefficient(self, order, lambda_list):
    """
        Calculate the coefficient of lagrange polynomial
        """
    assert order in [0, 1, 2, 3]
    assert order == len(lambda_list) - 1
    if order == 0:
        return [[1]]
    elif order == 1:
        return [[1 / (lambda_list[0] - lambda_list[1]), -lambda_list[1] / (
            lambda_list[0] - lambda_list[1])], [1 / (lambda_list[1] -
            lambda_list[0]), -lambda_list[0] / (lambda_list[1] -
            lambda_list[0])]]
    elif order == 2:
        denominator1 = (lambda_list[0] - lambda_list[1]) * (lambda_list[0] -
            lambda_list[2])
        denominator2 = (lambda_list[1] - lambda_list[0]) * (lambda_list[1] -
            lambda_list[2])
        denominator3 = (lambda_list[2] - lambda_list[0]) * (lambda_list[2] -
            lambda_list[1])
        return [[1 / denominator1, (-lambda_list[1] - lambda_list[2]) /
            denominator1, lambda_list[1] * lambda_list[2] / denominator1],
            [1 / denominator2, (-lambda_list[0] - lambda_list[2]) /
            denominator2, lambda_list[0] * lambda_list[2] / denominator2],
            [1 / denominator3, (-lambda_list[0] - lambda_list[1]) /
            denominator3, lambda_list[0] * lambda_list[1] / denominator3]]
    elif order == 3:
        denominator1 = (lambda_list[0] - lambda_list[1]) * (lambda_list[0] -
            lambda_list[2]) * (lambda_list[0] - lambda_list[3])
        denominator2 = (lambda_list[1] - lambda_list[0]) * (lambda_list[1] -
            lambda_list[2]) * (lambda_list[1] - lambda_list[3])
        denominator3 = (lambda_list[2] - lambda_list[0]) * (lambda_list[2] -
            lambda_list[1]) * (lambda_list[2] - lambda_list[3])
        denominator4 = (lambda_list[3] - lambda_list[0]) * (lambda_list[3] -
            lambda_list[1]) * (lambda_list[3] - lambda_list[2])
        return [[1 / denominator1, (-lambda_list[1] - lambda_list[2] -
            lambda_list[3]) / denominator1, (lambda_list[1] * lambda_list[2
            ] + lambda_list[1] * lambda_list[3] + lambda_list[2] *
            lambda_list[3]) / denominator1, -lambda_list[1] * lambda_list[2
            ] * lambda_list[3] / denominator1], [1 / denominator2, (-
            lambda_list[0] - lambda_list[2] - lambda_list[3]) /
            denominator2, (lambda_list[0] * lambda_list[2] + lambda_list[0] *
            lambda_list[3] + lambda_list[2] * lambda_list[3]) /
            denominator2, -lambda_list[0] * lambda_list[2] * lambda_list[3] /
            denominator2], [1 / denominator3, (-lambda_list[0] -
            lambda_list[1] - lambda_list[3]) / denominator3, (lambda_list[0
            ] * lambda_list[1] + lambda_list[0] * lambda_list[3] + 
            lambda_list[1] * lambda_list[3]) / denominator3, -lambda_list[0
            ] * lambda_list[1] * lambda_list[3] / denominator3], [1 /
            denominator4, (-lambda_list[0] - lambda_list[1] - lambda_list[2
            ]) / denominator4, (lambda_list[0] * lambda_list[1] + 
            lambda_list[0] * lambda_list[2] + lambda_list[1] * lambda_list[
            2]) / denominator4, -lambda_list[0] * lambda_list[1] *
            lambda_list[2] / denominator4]]
