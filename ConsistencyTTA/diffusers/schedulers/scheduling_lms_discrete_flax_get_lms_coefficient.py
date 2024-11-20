def get_lms_coefficient(self, state: LMSDiscreteSchedulerState, order, t,
    current_order):
    """
        Compute a linear multistep coefficient.

        Args:
            order (TODO):
            t (TODO):
            current_order (TODO):
        """

    def lms_derivative(tau):
        prod = 1.0
        for k in range(order):
            if current_order == k:
                continue
            prod *= (tau - state.sigmas[t - k]) / (state.sigmas[t -
                current_order] - state.sigmas[t - k])
        return prod
    integrated_coeff = integrate.quad(lms_derivative, state.sigmas[t],
        state.sigmas[t + 1], epsrel=0.0001)[0]
    return integrated_coeff
