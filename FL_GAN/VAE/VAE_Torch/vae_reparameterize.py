def reparameterize(self, mu, sigma, noise=True):
    normal = dist.normal.Normal(0, 1)
    eps = normal.sample(sigma.shape).cuda()
    return mu + eps * sigma if noise else mu
