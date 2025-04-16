def get_dummy_components(self):
    torch.manual_seed(0)
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=
        False)
    torch.manual_seed(0)
    unet = I2VGenXLUNet(block_out_channels=(4, 8), layers_per_block=1,
        sample_size=32, in_channels=4, out_channels=4, down_block_types=(
        'CrossAttnDownBlock3D', 'DownBlock3D'), up_block_types=('UpBlock3D',
        'CrossAttnUpBlock3D'), cross_attention_dim=4, attention_head_dim=4,
        num_attention_heads=None, norm_num_groups=2)
    torch.manual_seed(0)
    vae = AutoencoderKL(block_out_channels=(8,), in_channels=3,
        out_channels=3, down_block_types=['DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D'], latent_channels=4, sample_size
        =32, norm_num_groups=2)
    torch.manual_seed(0)
    text_encoder_config = CLIPTextConfig(bos_token_id=0, eos_token_id=2,
        hidden_size=4, intermediate_size=16, layer_norm_eps=1e-05,
        num_attention_heads=2, num_hidden_layers=2, pad_token_id=1,
        vocab_size=1000, hidden_act='gelu', projection_dim=32)
    text_encoder = CLIPTextModel(text_encoder_config)
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    torch.manual_seed(0)
    vision_encoder_config = CLIPVisionConfig(hidden_size=4, projection_dim=
        4, num_hidden_layers=2, num_attention_heads=2, image_size=32,
        intermediate_size=16, patch_size=1)
    image_encoder = CLIPVisionModelWithProjection(vision_encoder_config)
    torch.manual_seed(0)
    feature_extractor = CLIPImageProcessor(crop_size=32, size=32)
    components = {'unet': unet, 'scheduler': scheduler, 'vae': vae,
        'text_encoder': text_encoder, 'image_encoder': image_encoder,
        'tokenizer': tokenizer, 'feature_extractor': feature_extractor}
    return components
