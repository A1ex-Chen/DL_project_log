def get_timestep_ratio_conditioning(self, t, alphas_cumprod):
    s = torch.tensor([0.003])
    clamp_range = [0, 1]
    min_var = torch.cos(s / (1 + s) * torch.pi * 0.5) ** 2
    var = alphas_cumprod[t]
    var = var.clamp(*clamp_range)
    s, min_var = s.to(var.device), min_var.to(var.device)
    ratio = ((var * min_var) ** 0.5).acos() / (torch.pi * 0.5) * (1 + s) - s
    return ratio
