def calculate_loss(self, batch):
    input_x = torch.stack(batch)
    recon_x, mu, logvar = self.model(input_x)
    CE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * input_x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1))
    return CE + self.beta * KLD
