def lms_derivative(tau):
    prod = 1.0
    for k in range(order):
        if current_order == k:
            continue
        prod *= (tau - state.sigmas[t - k]) / (state.sigmas[t -
            current_order] - state.sigmas[t - k])
    return prod
