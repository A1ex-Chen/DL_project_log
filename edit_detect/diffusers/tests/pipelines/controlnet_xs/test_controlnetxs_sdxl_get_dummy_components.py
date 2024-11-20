def get_dummy_components(self):
    torch.manual_seed(0)
    unet = UNet2DConditionModel(block_out_channels=(4, 8), layers_per_block
        =2, sample_size=16, in_channels=4, out_channels=4, down_block_types
        =('DownBlock2D', 'CrossAttnDownBlock2D'), up_block_types=(
        'CrossAttnUpBlock2D', 'UpBlock2D'), use_linear_projection=True,
        norm_num_groups=4, attention_head_dim=(2, 4), addition_embed_type=
        'text_time', addition_time_embed_dim=8,
        transformer_layers_per_block=(1, 2),
        projection_class_embeddings_input_dim=56, cross_attention_dim=8)
    torch.manual_seed(0)
    controlnet = ControlNetXSAdapter.from_unet(unet=unet, size_ratio=0.5,
        learn_time_embedding=True, conditioning_embedding_out_channels=(2, 2))
    torch.manual_seed(0)
    scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012,
        steps_offset=1, beta_schedule='scaled_linear', timestep_spacing=
        'leading')
    torch.manual_seed(0)
    vae = AutoencoderKL(block_out_channels=[4, 8], in_channels=3,
        out_channels=3, down_block_types=['DownEncoderBlock2D',
        'DownEncoderBlock2D'], up_block_types=['UpDecoderBlock2D',
        'UpDecoderBlock2D'], latent_channels=4, norm_num_groups=2)
    torch.manual_seed(0)
    text_encoder_config = CLIPTextConfig(bos_token_id=0, eos_token_id=2,
        hidden_size=4, intermediate_size=37, layer_norm_eps=1e-05,
        num_attention_heads=4, num_hidden_layers=5, pad_token_id=1,
        vocab_size=1000, hidden_act='gelu', projection_dim=8)
    text_encoder = CLIPTextModel(text_encoder_config)
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    components = {'unet': unet, 'controlnet': controlnet, 'scheduler':
        scheduler, 'vae': vae, 'text_encoder': text_encoder, 'tokenizer':
        tokenizer, 'text_encoder_2': text_encoder_2, 'tokenizer_2':
        tokenizer_2, 'feature_extractor': None}
    return components
