def reset_parameters(self, gain=1):
    """
        reset_parameters()
        """
    stdev = 1.0 / math.sqrt(self.hidden_size)
    for param in self.parameters():
        param.data.uniform_(-stdev, stdev)
