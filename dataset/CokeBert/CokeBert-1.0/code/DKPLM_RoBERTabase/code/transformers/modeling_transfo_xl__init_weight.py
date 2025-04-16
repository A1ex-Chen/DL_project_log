def _init_weight(self, weight):
    if self.config.init == 'uniform':
        nn.init.uniform_(weight, -self.config.init_range, self.config.
            init_range)
    elif self.config.init == 'normal':
        nn.init.normal_(weight, 0.0, self.config.init_std)
