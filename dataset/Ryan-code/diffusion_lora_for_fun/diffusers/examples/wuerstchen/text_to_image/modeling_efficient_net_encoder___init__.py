@register_to_config
def __init__(self, c_latent=16, c_cond=1280, effnet='efficientnet_v2_s'):
    super().__init__()
    if effnet == 'efficientnet_v2_s':
        self.backbone = efficientnet_v2_s(weights='DEFAULT').features
    else:
        self.backbone = efficientnet_v2_l(weights='DEFAULT').features
    self.mapper = nn.Sequential(nn.Conv2d(c_cond, c_latent, kernel_size=1,
        bias=False), nn.BatchNorm2d(c_latent))
