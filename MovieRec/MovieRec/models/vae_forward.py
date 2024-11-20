def forward(self, x):
    x = F.normalize(x)
    x = self.input_dropout(x)
    for i, layer in enumerate(self.encoder):
        x = layer(x)
        if i != len(self.encoder) - 1:
            x = torch.tanh(x)
    mu, logvar = x[:, :self.latent_dim], x[:, self.latent_dim:]
    if self.training:
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        x = mu + eps * sigma
    else:
        x = mu
    for i, layer in enumerate(self.decoder):
        x = layer(x)
        if i != len(self.decoder) - 1:
            x = torch.tanh(x)
    return x, mu, logvar
