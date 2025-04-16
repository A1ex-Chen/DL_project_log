def reset_parameters(self, init_weight):
    """
        Sets initial random values for trainable parameters.
        """
    stdv = 1.0 / math.sqrt(self.num_units)
    self.linear_att.data.uniform_(-init_weight, init_weight)
    if self.normalize:
        self.normalize_scalar.data.fill_(stdv)
        self.normalize_bias.data.zero_()
