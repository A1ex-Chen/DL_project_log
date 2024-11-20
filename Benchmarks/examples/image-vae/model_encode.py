def encode(self, x):
    self.mu, self.log_v = self.encoder(x)
    self.mu, self.log_v = self.transformer(self.mu, self.log_v)
    std = self.log_v.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    y = eps.mul(std).add_(self.mu)
    return y
