@register_to_config
def __init__(self, in_channels: int=3, out_channels: int=3,
    down_block_types: Tuple[str, ...]=('DownEncoderBlock2D',),
    down_block_out_channels: Tuple[int, ...]=(64,), layers_per_down_block:
    int=1, up_block_types: Tuple[str, ...]=('UpDecoderBlock2D',),
    up_block_out_channels: Tuple[int, ...]=(64,), layers_per_up_block: int=
    1, act_fn: str='silu', latent_channels: int=4, norm_num_groups: int=32,
    sample_size: int=32, scaling_factor: float=0.18215) ->None:
    super().__init__()
    self.encoder = Encoder(in_channels=in_channels, out_channels=
        latent_channels, down_block_types=down_block_types,
        block_out_channels=down_block_out_channels, layers_per_block=
        layers_per_down_block, act_fn=act_fn, norm_num_groups=
        norm_num_groups, double_z=True)
    self.decoder = MaskConditionDecoder(in_channels=latent_channels,
        out_channels=out_channels, up_block_types=up_block_types,
        block_out_channels=up_block_out_channels, layers_per_block=
        layers_per_up_block, act_fn=act_fn, norm_num_groups=norm_num_groups)
    self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
    self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
    self.use_slicing = False
    self.use_tiling = False
    self.register_to_config(block_out_channels=up_block_out_channels)
    self.register_to_config(force_upcast=False)
