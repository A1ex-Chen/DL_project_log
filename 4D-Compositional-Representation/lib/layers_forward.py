def forward(self, x, c):
    batch_size = x.size(0)
    gamma = self.fc_gamma(c)
    beta = self.fc_beta(c)
    gamma = gamma.view(batch_size, self.f_dim, 1)
    beta = beta.view(batch_size, self.f_dim, 1)
    net = self.bn(x)
    out = gamma * net + beta
    return out
