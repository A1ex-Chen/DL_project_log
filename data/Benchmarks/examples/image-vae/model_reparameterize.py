def reparameterize(self, mu, logvar):
    if self.training:
        std = torch.exp(logvar.mul(0.5))
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu
