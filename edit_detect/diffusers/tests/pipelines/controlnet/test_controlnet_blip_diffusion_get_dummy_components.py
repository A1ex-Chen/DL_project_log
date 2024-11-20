def get_dummy_components(self):
    torch.manual_seed(0)
    text_encoder_config = CLIPTextConfig(vocab_size=1000, hidden_size=16,
        intermediate_size=16, projection_dim=16, num_hidden_layers=1,
        num_attention_heads=1, max_position_embeddings=77)
    text_encoder = ContextCLIPTextModel(text_encoder_config)
    vae = AutoencoderKL(in_channels=4, out_channels=4, down_block_types=(
        'DownEncoderBlock2D',), up_block_types=('UpDecoderBlock2D',),
        block_out_channels=(32,), layers_per_block=1, act_fn='silu',
        latent_channels=4, norm_num_groups=16, sample_size=16)
    blip_vision_config = {'hidden_size': 16, 'intermediate_size': 16,
        'num_hidden_layers': 1, 'num_attention_heads': 1, 'image_size': 224,
        'patch_size': 14, 'hidden_act': 'quick_gelu'}
    blip_qformer_config = {'vocab_size': 1000, 'hidden_size': 16,
        'num_hidden_layers': 1, 'num_attention_heads': 1,
        'intermediate_size': 16, 'max_position_embeddings': 512,
        'cross_attention_frequency': 1, 'encoder_hidden_size': 16}
    qformer_config = Blip2Config(vision_config=blip_vision_config,
        qformer_config=blip_qformer_config, num_query_tokens=16, tokenizer=
        'hf-internal-testing/tiny-random-bert')
    qformer = Blip2QFormerModel(qformer_config)
    unet = UNet2DConditionModel(block_out_channels=(4, 16),
        layers_per_block=1, norm_num_groups=4, sample_size=16, in_channels=
        4, out_channels=4, down_block_types=('DownBlock2D',
        'CrossAttnDownBlock2D'), up_block_types=('CrossAttnUpBlock2D',
        'UpBlock2D'), cross_attention_dim=16)
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', set_alpha_to_one=False,
        skip_prk_steps=True)
    controlnet = ControlNetModel(block_out_channels=(4, 16),
        layers_per_block=1, in_channels=4, norm_num_groups=4,
        down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
        cross_attention_dim=16, conditioning_embedding_out_channels=(8, 16))
    vae.eval()
    qformer.eval()
    text_encoder.eval()
    image_processor = BlipImageProcessor()
    components = {'text_encoder': text_encoder, 'vae': vae, 'qformer':
        qformer, 'unet': unet, 'tokenizer': tokenizer, 'scheduler':
        scheduler, 'controlnet': controlnet, 'image_processor': image_processor
        }
    return components
