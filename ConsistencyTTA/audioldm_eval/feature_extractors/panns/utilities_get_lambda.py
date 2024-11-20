def get_lambda(self, batch_size):
    """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
    mixup_lambdas = []
    for n in range(0, batch_size, 2):
        lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
        mixup_lambdas.append(lam)
        mixup_lambdas.append(1.0 - lam)
    return np.array(mixup_lambdas)
