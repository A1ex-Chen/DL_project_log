def get_lms_coefficient(self, order, t, current_order):
    """
        Compute the linear multistep coefficient.

        Args:
            order ():
            t ():
            current_order ():
        """

    def lms_derivative(tau):
        prod = 1.0
        for k in range(order):
            if current_order == k:
                continue
            prod *= (tau - self.sigmas[t - k]) / (self.sigmas[t -
                current_order] - self.sigmas[t - k])
        return prod
    integrated_coeff = integrate.quad(lms_derivative, self.sigmas[t], self.
        sigmas[t + 1], epsrel=0.0001)[0]
    return integrated_coeff
