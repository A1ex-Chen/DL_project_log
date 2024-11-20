def sample(self):
    x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.
        parameters.device)
    return x
