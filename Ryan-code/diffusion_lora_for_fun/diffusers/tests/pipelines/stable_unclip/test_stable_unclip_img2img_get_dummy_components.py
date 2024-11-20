def get_dummy_components(self):
    embedder_hidden_size = 32
    embedder_projection_dim = embedder_hidden_size
    feature_extractor = CLIPImageProcessor(crop_size=32, size=32)
    torch.manual_seed(0)
    image_encoder = CLIPVisionModelWithProjection(CLIPVisionConfig(
        hidden_size=embedder_hidden_size, projection_dim=
        embedder_projection_dim, num_hidden_layers=5, num_attention_heads=4,
        image_size=32, intermediate_size=37, patch_size=1))
    torch.manual_seed(0)
    image_normalizer = StableUnCLIPImageNormalizer(embedding_dim=
        embedder_hidden_size)
    image_noising_scheduler = DDPMScheduler(beta_schedule='squaredcos_cap_v2')
    torch.manual_seed(0)
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    torch.manual_seed(0)
    text_encoder = CLIPTextModel(CLIPTextConfig(bos_token_id=0,
        eos_token_id=2, hidden_size=embedder_hidden_size, projection_dim=32,
        intermediate_size=37, layer_norm_eps=1e-05, num_attention_heads=4,
        num_hidden_layers=5, pad_token_id=1, vocab_size=1000))
    torch.manual_seed(0)
    unet = UNet2DConditionModel(sample_size=32, in_channels=4, out_channels
        =4, down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
        up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'),
        block_out_channels=(32, 64), attention_head_dim=(2, 4),
        class_embed_type='projection',
        projection_class_embeddings_input_dim=embedder_projection_dim * 2,
        cross_attention_dim=embedder_hidden_size, layers_per_block=1,
        upcast_attention=True, use_linear_projection=True)
    torch.manual_seed(0)
    scheduler = DDIMScheduler(beta_schedule='scaled_linear', beta_start=
        0.00085, beta_end=0.012, prediction_type='v_prediction',
        set_alpha_to_one=False, steps_offset=1)
    torch.manual_seed(0)
    vae = AutoencoderKL()
    components = {'feature_extractor': feature_extractor, 'image_encoder':
        image_encoder.eval(), 'image_normalizer': image_normalizer.eval(),
        'image_noising_scheduler': image_noising_scheduler, 'tokenizer':
        tokenizer, 'text_encoder': text_encoder.eval(), 'unet': unet.eval(),
        'scheduler': scheduler, 'vae': vae.eval()}
    return components
