def lms_derivative(tau):
    prod = 1.0
    for k in range(order):
        if current_order == k:
            continue
        prod *= (tau - self.sigmas[t - k]) / (self.sigmas[t - current_order
            ] - self.sigmas[t - k])
    return prod
