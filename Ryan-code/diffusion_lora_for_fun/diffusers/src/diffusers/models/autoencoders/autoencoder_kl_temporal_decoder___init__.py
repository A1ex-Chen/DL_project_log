@register_to_config
def __init__(self, in_channels: int=3, out_channels: int=3,
    down_block_types: Tuple[str]=('DownEncoderBlock2D',),
    block_out_channels: Tuple[int]=(64,), layers_per_block: int=1,
    latent_channels: int=4, sample_size: int=32, scaling_factor: float=
    0.18215, force_upcast: float=True):
    super().__init__()
    self.encoder = Encoder(in_channels=in_channels, out_channels=
        latent_channels, down_block_types=down_block_types,
        block_out_channels=block_out_channels, layers_per_block=
        layers_per_block, double_z=True)
    self.decoder = TemporalDecoder(in_channels=latent_channels,
        out_channels=out_channels, block_out_channels=block_out_channels,
        layers_per_block=layers_per_block)
    self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
    sample_size = self.config.sample_size[0] if isinstance(self.config.
        sample_size, (list, tuple)) else self.config.sample_size
    self.tile_latent_min_size = int(sample_size / 2 ** (len(self.config.
        block_out_channels) - 1))
    self.tile_overlap_factor = 0.25
