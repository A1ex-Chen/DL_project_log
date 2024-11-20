def set_weights(self, weights):
    params = self.weights
    num_vars = int((len(params) - 1) / 2)
    if len(weights) == 3 * num_vars + 1:
        weights = weights[:len(params)]
    super().set_weights(weights)
