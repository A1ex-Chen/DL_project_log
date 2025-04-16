@register_to_config
def __init__(self, in_channels: int=3, out_channels: int=3,
    encoder_block_out_channels: Tuple[int, ...]=(64, 64, 64, 64),
    decoder_block_out_channels: Tuple[int, ...]=(64, 64, 64, 64), act_fn:
    str='relu', upsample_fn: str='nearest', latent_channels: int=4,
    upsampling_scaling_factor: int=2, num_encoder_blocks: Tuple[int, ...]=(
    1, 3, 3, 3), num_decoder_blocks: Tuple[int, ...]=(3, 3, 3, 1),
    latent_magnitude: int=3, latent_shift: float=0.5, force_upcast: bool=
    False, scaling_factor: float=1.0):
    super().__init__()
    if len(encoder_block_out_channels) != len(num_encoder_blocks):
        raise ValueError(
            '`encoder_block_out_channels` should have the same length as `num_encoder_blocks`.'
            )
    if len(decoder_block_out_channels) != len(num_decoder_blocks):
        raise ValueError(
            '`decoder_block_out_channels` should have the same length as `num_decoder_blocks`.'
            )
    self.encoder = EncoderTiny(in_channels=in_channels, out_channels=
        latent_channels, num_blocks=num_encoder_blocks, block_out_channels=
        encoder_block_out_channels, act_fn=act_fn)
    self.decoder = DecoderTiny(in_channels=latent_channels, out_channels=
        out_channels, num_blocks=num_decoder_blocks, block_out_channels=
        decoder_block_out_channels, upsampling_scaling_factor=
        upsampling_scaling_factor, act_fn=act_fn, upsample_fn=upsample_fn)
    self.latent_magnitude = latent_magnitude
    self.latent_shift = latent_shift
    self.scaling_factor = scaling_factor
    self.use_slicing = False
    self.use_tiling = False
    self.spatial_scale_factor = 2 ** out_channels
    self.tile_overlap_factor = 0.125
    self.tile_sample_min_size = 512
    self.tile_latent_min_size = (self.tile_sample_min_size // self.
        spatial_scale_factor)
    self.register_to_config(block_out_channels=decoder_block_out_channels)
    self.register_to_config(force_upcast=False)
