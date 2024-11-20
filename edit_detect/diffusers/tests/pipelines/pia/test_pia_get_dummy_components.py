def get_dummy_components(self):
    torch.manual_seed(0)
    unet = UNet2DConditionModel(block_out_channels=(32, 64),
        layers_per_block=2, sample_size=32, in_channels=4, out_channels=4,
        down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'),
        cross_attention_dim=32, norm_num_groups=2)
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='linear', clip_sample=False)
    torch.manual_seed(0)
    vae = AutoencoderKL(block_out_channels=[32, 64], in_channels=3,
        out_channels=3, down_block_types=['DownEncoderBlock2D',
        'DownEncoderBlock2D'], up_block_types=['UpDecoderBlock2D',
        'UpDecoderBlock2D'], latent_channels=4)
    torch.manual_seed(0)
    text_encoder_config = CLIPTextConfig(bos_token_id=0, eos_token_id=2,
        hidden_size=32, intermediate_size=37, layer_norm_eps=1e-05,
        num_attention_heads=4, num_hidden_layers=5, pad_token_id=1,
        vocab_size=1000)
    text_encoder = CLIPTextModel(text_encoder_config)
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    motion_adapter = MotionAdapter(block_out_channels=(32, 64),
        motion_layers_per_block=2, motion_norm_num_groups=2,
        motion_num_attention_heads=4, conv_in_channels=9)
    components = {'unet': unet, 'scheduler': scheduler, 'vae': vae,
        'motion_adapter': motion_adapter, 'text_encoder': text_encoder,
        'tokenizer': tokenizer, 'feature_extractor': None, 'image_encoder':
        None}
    return components
