def get_dummy_components(self, time_cond_proj_dim=None):
    torch.manual_seed(0)
    unet = UNet2DConditionModel(block_out_channels=(32, 64, 128),
        layers_per_block=2, time_cond_proj_dim=time_cond_proj_dim,
        sample_size=32, in_channels=4, out_channels=4, down_block_types=(
        'DownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'CrossAttnUpBlock2D',
        'UpBlock2D'), attention_head_dim=(2, 4, 8), use_linear_projection=
        True, addition_embed_type='text_time', addition_time_embed_dim=8,
        transformer_layers_per_block=(1, 2, 4),
        projection_class_embeddings_input_dim=80, cross_attention_dim=64,
        norm_num_groups=1)
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='linear', clip_sample=False)
    torch.manual_seed(0)
    vae = AutoencoderKL(block_out_channels=[32, 64], in_channels=3,
        out_channels=3, down_block_types=['DownEncoderBlock2D',
        'DownEncoderBlock2D'], up_block_types=['UpDecoderBlock2D',
        'UpDecoderBlock2D'], latent_channels=4, sample_size=128)
    torch.manual_seed(0)
    text_encoder_config = CLIPTextConfig(bos_token_id=0, eos_token_id=2,
        hidden_size=32, intermediate_size=37, layer_norm_eps=1e-05,
        num_attention_heads=4, num_hidden_layers=5, pad_token_id=1,
        vocab_size=1000, hidden_act='gelu', projection_dim=32)
    text_encoder = CLIPTextModel(text_encoder_config)
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    motion_adapter = MotionAdapter(block_out_channels=(32, 64, 128),
        motion_layers_per_block=2, motion_norm_num_groups=2,
        motion_num_attention_heads=4, use_motion_mid_block=False)
    components = {'unet': unet, 'scheduler': scheduler, 'vae': vae,
        'motion_adapter': motion_adapter, 'text_encoder': text_encoder,
        'tokenizer': tokenizer, 'text_encoder_2': text_encoder_2,
        'tokenizer_2': tokenizer_2, 'feature_extractor': None,
        'image_encoder': None}
    return components
