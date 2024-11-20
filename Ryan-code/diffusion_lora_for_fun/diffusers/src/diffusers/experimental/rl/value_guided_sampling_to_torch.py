def to_torch(self, x_in):
    if isinstance(x_in, dict):
        return {k: self.to_torch(v) for k, v in x_in.items()}
    elif torch.is_tensor(x_in):
        return x_in.to(self.unet.device)
    return torch.tensor(x_in, device=self.unet.device)
