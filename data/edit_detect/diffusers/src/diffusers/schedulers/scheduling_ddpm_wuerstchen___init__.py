@register_to_config
def __init__(self, scaler: float=1.0, s: float=0.008):
    self.scaler = scaler
    self.s = torch.tensor([s])
    self._init_alpha_cumprod = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5
        ) ** 2
    self.init_noise_sigma = 1.0
