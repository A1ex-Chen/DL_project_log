def to(self, device: torch.device):
    return RotatedBoxes(self.tensor.to(device=device))
