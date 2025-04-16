@register_to_config
def __init__(self, in_channels: int=3, out_channels: int=3,
    up_down_scale_factor: int=2, levels: int=2, bottleneck_blocks: int=12,
    embed_dim: int=384, latent_channels: int=4, num_vq_embeddings: int=8192,
    scale_factor: float=0.3764):
    super().__init__()
    c_levels = [(embed_dim // 2 ** i) for i in reversed(range(levels))]
    self.in_block = nn.Sequential(nn.PixelUnshuffle(up_down_scale_factor),
        nn.Conv2d(in_channels * up_down_scale_factor ** 2, c_levels[0],
        kernel_size=1))
    down_blocks = []
    for i in range(levels):
        if i > 0:
            down_blocks.append(nn.Conv2d(c_levels[i - 1], c_levels[i],
                kernel_size=4, stride=2, padding=1))
        block = MixingResidualBlock(c_levels[i], c_levels[i] * 4)
        down_blocks.append(block)
    down_blocks.append(nn.Sequential(nn.Conv2d(c_levels[-1],
        latent_channels, kernel_size=1, bias=False), nn.BatchNorm2d(
        latent_channels)))
    self.down_blocks = nn.Sequential(*down_blocks)
    self.vquantizer = VectorQuantizer(num_vq_embeddings, vq_embed_dim=
        latent_channels, legacy=False, beta=0.25)
    up_blocks = [nn.Sequential(nn.Conv2d(latent_channels, c_levels[-1],
        kernel_size=1))]
    for i in range(levels):
        for j in range(bottleneck_blocks if i == 0 else 1):
            block = MixingResidualBlock(c_levels[levels - 1 - i], c_levels[
                levels - 1 - i] * 4)
            up_blocks.append(block)
        if i < levels - 1:
            up_blocks.append(nn.ConvTranspose2d(c_levels[levels - 1 - i],
                c_levels[levels - 2 - i], kernel_size=4, stride=2, padding=1))
    self.up_blocks = nn.Sequential(*up_blocks)
    self.out_block = nn.Sequential(nn.Conv2d(c_levels[0], out_channels * 
        up_down_scale_factor ** 2, kernel_size=1), nn.PixelShuffle(
        up_down_scale_factor))
