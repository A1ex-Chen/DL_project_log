def get_dummy_components(self, skip_first_text_encoder=False,
    time_cond_proj_dim=None):
    torch.manual_seed(0)
    unet = UNet2DConditionModel(block_out_channels=(32, 64),
        layers_per_block=2, sample_size=32, in_channels=4, out_channels=4,
        time_cond_proj_dim=time_cond_proj_dim, down_block_types=(
        'DownBlock2D', 'CrossAttnDownBlock2D'), up_block_types=(
        'CrossAttnUpBlock2D', 'UpBlock2D'), attention_head_dim=(2, 4),
        use_linear_projection=True, addition_embed_type='text_time',
        addition_time_embed_dim=8, transformer_layers_per_block=(1, 2),
        projection_class_embeddings_input_dim=80, cross_attention_dim=64 if
        not skip_first_text_encoder else 32)
    scheduler = DPMSolverMultistepScheduler(algorithm_type=
        'sde-dpmsolver++', solver_order=2)
    torch.manual_seed(0)
    vae = AutoencoderKL(block_out_channels=[32, 64], in_channels=3,
        out_channels=3, down_block_types=['DownEncoderBlock2D',
        'DownEncoderBlock2D'], up_block_types=['UpDecoderBlock2D',
        'UpDecoderBlock2D'], latent_channels=4, sample_size=128)
    torch.manual_seed(0)
    image_encoder_config = CLIPVisionConfig(hidden_size=32, image_size=224,
        projection_dim=32, intermediate_size=37, num_attention_heads=4,
        num_channels=3, num_hidden_layers=5, patch_size=14)
    image_encoder = CLIPVisionModelWithProjection(image_encoder_config)
    feature_extractor = CLIPImageProcessor(crop_size=224, do_center_crop=
        True, do_normalize=True, do_resize=True, image_mean=[0.48145466, 
        0.4578275, 0.40821073], image_std=[0.26862954, 0.26130258, 
        0.27577711], resample=3, size=224)
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
    components = {'unet': unet, 'scheduler': scheduler, 'vae': vae,
        'text_encoder': text_encoder if not skip_first_text_encoder else
        None, 'tokenizer': tokenizer if not skip_first_text_encoder else
        None, 'text_encoder_2': text_encoder_2, 'tokenizer_2': tokenizer_2,
        'image_encoder': image_encoder, 'feature_extractor': feature_extractor}
    return components
