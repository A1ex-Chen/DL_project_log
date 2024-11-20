def to(self, device: torch.device) ->'ROIMasks':
    return ROIMasks(self.tensor.to(device))
