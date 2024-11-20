def get_dummy_components(self):
    torch.manual_seed(0)
    unet = UNetSpatioTemporalConditionModel(block_out_channels=(32, 64),
        layers_per_block=2, sample_size=32, in_channels=8, out_channels=4,
        down_block_types=('CrossAttnDownBlockSpatioTemporal',
        'DownBlockSpatioTemporal'), up_block_types=('UpBlockSpatioTemporal',
        'CrossAttnUpBlockSpatioTemporal'), cross_attention_dim=32,
        num_attention_heads=8, projection_class_embeddings_input_dim=96,
        addition_time_embed_dim=32)
    scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', interpolation_type='linear',
        num_train_timesteps=1000, prediction_type='v_prediction', sigma_max
        =700.0, sigma_min=0.002, steps_offset=1, timestep_spacing='leading',
        timestep_type='continuous', trained_betas=None, use_karras_sigmas=True)
    torch.manual_seed(0)
    vae = AutoencoderKLTemporalDecoder(block_out_channels=[32, 64],
        in_channels=3, out_channels=3, down_block_types=[
        'DownEncoderBlock2D', 'DownEncoderBlock2D'], latent_channels=4)
    torch.manual_seed(0)
    config = CLIPVisionConfig(hidden_size=32, projection_dim=32,
        num_hidden_layers=5, num_attention_heads=4, image_size=32,
        intermediate_size=37, patch_size=1)
    image_encoder = CLIPVisionModelWithProjection(config)
    torch.manual_seed(0)
    feature_extractor = CLIPImageProcessor(crop_size=32, size=32)
    components = {'unet': unet, 'image_encoder': image_encoder, 'scheduler':
        scheduler, 'vae': vae, 'feature_extractor': feature_extractor}
    return components
