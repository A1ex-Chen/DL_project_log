def get_decay(self, optimization_step):
    """
        Compute the decay factor for the exponential moving average.
        """
    value = (1 + optimization_step) / (10 + optimization_step)
    return 1 - min(self.decay, value)
