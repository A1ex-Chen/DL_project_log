def sampling(self):
    z = torch.randn(64, self.n_latent_features).to(self.device)
    z = self.fc3(z)
    return self.decoder(z)
