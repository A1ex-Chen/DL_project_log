@register_to_config
def __init__(self, scaling_factor: float=0.18215, latent_channels: int=4,
    sample_size: int=32, encoder_act_fn: str='silu',
    encoder_block_out_channels: Tuple[int, ...]=(128, 256, 512, 512),
    encoder_double_z: bool=True, encoder_down_block_types: Tuple[str, ...]=
    ('DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D',
    'DownEncoderBlock2D'), encoder_in_channels: int=3,
    encoder_layers_per_block: int=2, encoder_norm_num_groups: int=32,
    encoder_out_channels: int=4, decoder_add_attention: bool=False,
    decoder_block_out_channels: Tuple[int, ...]=(320, 640, 1024, 1024),
    decoder_down_block_types: Tuple[str, ...]=('ResnetDownsampleBlock2D',
    'ResnetDownsampleBlock2D', 'ResnetDownsampleBlock2D',
    'ResnetDownsampleBlock2D'), decoder_downsample_padding: int=1,
    decoder_in_channels: int=7, decoder_layers_per_block: int=3,
    decoder_norm_eps: float=1e-05, decoder_norm_num_groups: int=32,
    decoder_num_train_timesteps: int=1024, decoder_out_channels: int=6,
    decoder_resnet_time_scale_shift: str='scale_shift',
    decoder_time_embedding_type: str='learned', decoder_up_block_types:
    Tuple[str, ...]=('ResnetUpsampleBlock2D', 'ResnetUpsampleBlock2D',
    'ResnetUpsampleBlock2D', 'ResnetUpsampleBlock2D')):
    super().__init__()
    self.encoder = Encoder(act_fn=encoder_act_fn, block_out_channels=
        encoder_block_out_channels, double_z=encoder_double_z,
        down_block_types=encoder_down_block_types, in_channels=
        encoder_in_channels, layers_per_block=encoder_layers_per_block,
        norm_num_groups=encoder_norm_num_groups, out_channels=
        encoder_out_channels)
    self.decoder_unet = UNet2DModel(add_attention=decoder_add_attention,
        block_out_channels=decoder_block_out_channels, down_block_types=
        decoder_down_block_types, downsample_padding=
        decoder_downsample_padding, in_channels=decoder_in_channels,
        layers_per_block=decoder_layers_per_block, norm_eps=
        decoder_norm_eps, norm_num_groups=decoder_norm_num_groups,
        num_train_timesteps=decoder_num_train_timesteps, out_channels=
        decoder_out_channels, resnet_time_scale_shift=
        decoder_resnet_time_scale_shift, time_embedding_type=
        decoder_time_embedding_type, up_block_types=decoder_up_block_types)
    self.decoder_scheduler = ConsistencyDecoderScheduler()
    self.register_to_config(block_out_channels=encoder_block_out_channels)
    self.register_to_config(force_upcast=False)
    self.register_buffer('means', torch.tensor([0.38862467, 0.02253063, 
        0.07381133, -0.0171294])[None, :, None, None], persistent=False)
    self.register_buffer('stds', torch.tensor([0.9654121, 1.0440036, 
        0.76147926, 0.77022034])[None, :, None, None], persistent=False)
    self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
    self.use_slicing = False
    self.use_tiling = False
    self.tile_sample_min_size = self.config.sample_size
    sample_size = self.config.sample_size[0] if isinstance(self.config.
        sample_size, (list, tuple)) else self.config.sample_size
    self.tile_latent_min_size = int(sample_size / 2 ** (len(self.config.
        block_out_channels) - 1))
    self.tile_overlap_factor = 0.25
