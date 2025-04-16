def calculate_loss(self, batch):
    input_x = torch.stack(batch)
    recon_x = self.model(input_x)
    CE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * input_x, -1))
    return CE
