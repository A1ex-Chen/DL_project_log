def get_dummy_components(self):
    torch.manual_seed(0)
    unet = UNet2DConditionModel(block_out_channels=(32, 64),
        layers_per_block=2, sample_size=32, in_channels=9, out_channels=4,
        down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'),
        cross_attention_dim=32)
    scheduler = PNDMScheduler(skip_prk_steps=True)
    torch.manual_seed(0)
    vae = AutoencoderKL(block_out_channels=[32, 64], in_channels=3,
        out_channels=3, down_block_types=['DownEncoderBlock2D',
        'DownEncoderBlock2D'], up_block_types=['UpDecoderBlock2D',
        'UpDecoderBlock2D'], latent_channels=4)
    torch.manual_seed(0)
    config = CLIPVisionConfig(hidden_size=32, projection_dim=32,
        intermediate_size=37, layer_norm_eps=1e-05, num_attention_heads=4,
        num_hidden_layers=5, image_size=32, patch_size=4)
    image_encoder = PaintByExampleImageEncoder(config, proj_size=32)
    feature_extractor = CLIPImageProcessor(crop_size=32, size=32)
    components = {'unet': unet, 'scheduler': scheduler, 'vae': vae,
        'image_encoder': image_encoder, 'safety_checker': None,
        'feature_extractor': feature_extractor}
    return components
