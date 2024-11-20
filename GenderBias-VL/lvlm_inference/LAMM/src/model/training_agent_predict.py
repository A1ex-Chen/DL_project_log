@torch.no_grad()
def predict(self, batch):
    self.model.eval()
    string = self.model.generate_one_sample(batch)
    return string
