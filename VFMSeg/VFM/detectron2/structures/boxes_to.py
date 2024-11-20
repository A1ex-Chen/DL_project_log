def to(self, device: torch.device):
    return Boxes(self.tensor.to(device=device))
