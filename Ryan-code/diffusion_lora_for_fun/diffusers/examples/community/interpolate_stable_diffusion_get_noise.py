def get_noise(self, seed, dtype=torch.float32, height=512, width=512):
    """Takes in random seed and returns corresponding noise vector"""
    return torch.randn((1, self.unet.config.in_channels, height // 8, width //
        8), generator=torch.Generator(device=self.device).manual_seed(seed),
        device=self.device, dtype=dtype)
