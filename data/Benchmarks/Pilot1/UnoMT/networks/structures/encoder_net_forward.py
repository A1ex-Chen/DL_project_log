def forward(self, samples):
    if self.decoder is None:
        return self.encoder(samples)
    else:
        return self.decoder(self.encoder(samples))
