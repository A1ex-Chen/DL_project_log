def get_coefficients_fn(self, order, interval_start, interval_end,
    lambda_list, tau):
    assert order in [1, 2, 3, 4]
    assert order == len(lambda_list
        ), 'the length of lambda list must be equal to the order'
    coefficients = []
    lagrange_coefficient = self.lagrange_polynomial_coefficient(order - 1,
        lambda_list)
    for i in range(order):
        coefficient = 0
        for j in range(order):
            if self.predict_x0:
                coefficient += lagrange_coefficient[i][j
                    ] * self.get_coefficients_exponential_positive(order - 
                    1 - j, interval_start, interval_end, tau)
            else:
                coefficient += lagrange_coefficient[i][j
                    ] * self.get_coefficients_exponential_negative(order - 
                    1 - j, interval_start, interval_end)
        coefficients.append(coefficient)
    assert len(coefficients
        ) == order, 'the length of coefficients does not match the order'
    return coefficients
