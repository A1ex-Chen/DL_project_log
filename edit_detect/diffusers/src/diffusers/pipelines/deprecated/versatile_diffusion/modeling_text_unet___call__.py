def __call__(self, x):
    x = self.freq_bands * x.unsqueeze(-1)
    return torch.stack((x.sin(), x.cos()), dim=-1).permute(0, 1, 3, 4, 2
        ).reshape(*x.shape[:2], -1)
