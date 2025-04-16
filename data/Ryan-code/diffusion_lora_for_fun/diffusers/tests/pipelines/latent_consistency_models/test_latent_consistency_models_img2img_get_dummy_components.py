def get_dummy_components(self):
    torch.manual_seed(0)
    unet = UNet2DConditionModel(block_out_channels=(4, 8), layers_per_block
        =1, sample_size=32, in_channels=4, out_channels=4, down_block_types
        =('DownBlock2D', 'CrossAttnDownBlock2D'), up_block_types=(
        'CrossAttnUpBlock2D', 'UpBlock2D'), cross_attention_dim=32,
        norm_num_groups=2, time_cond_proj_dim=32)
    scheduler = LCMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=
        False)
    torch.manual_seed(0)
    vae = AutoencoderKL(block_out_channels=[4, 8], in_channels=3,
        out_channels=3, down_block_types=['DownEncoderBlock2D',
        'DownEncoderBlock2D'], up_block_types=['UpDecoderBlock2D',
        'UpDecoderBlock2D'], latent_channels=4, norm_num_groups=2)
    torch.manual_seed(0)
    text_encoder_config = CLIPTextConfig(bos_token_id=0, eos_token_id=2,
        hidden_size=32, intermediate_size=64, layer_norm_eps=1e-05,
        num_attention_heads=8, num_hidden_layers=3, pad_token_id=1,
        vocab_size=1000)
    text_encoder = CLIPTextModel(text_encoder_config)
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    components = {'unet': unet, 'scheduler': scheduler, 'vae': vae,
        'text_encoder': text_encoder, 'tokenizer': tokenizer,
        'safety_checker': None, 'feature_extractor': None, 'image_encoder':
        None, 'requires_safety_checker': False}
    return components
