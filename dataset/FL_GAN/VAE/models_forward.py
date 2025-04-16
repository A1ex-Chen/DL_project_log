def forward(self, x):
    h = self.encoder(x)
    z, mu, logvar = self._bottleneck(h)
    z = self.fc3(z)
    d = self.decoder(z)
    return d, mu, logvar
