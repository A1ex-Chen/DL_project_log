def __init__additional__(self):
    self.unet.config.in_channels = 4
    self.vae_scale_factor = 8
