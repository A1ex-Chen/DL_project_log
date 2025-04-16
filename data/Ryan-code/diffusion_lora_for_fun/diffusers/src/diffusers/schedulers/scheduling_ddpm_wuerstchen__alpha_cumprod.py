def _alpha_cumprod(self, t, device):
    if self.scaler > 1:
        t = 1 - (1 - t) ** self.scaler
    elif self.scaler < 1:
        t = t ** self.scaler
    alpha_cumprod = torch.cos((t + self.s.to(device)) / (1 + self.s.to(
        device)) * torch.pi * 0.5) ** 2 / self._init_alpha_cumprod.to(device)
    return alpha_cumprod.clamp(0.0001, 0.9999)
