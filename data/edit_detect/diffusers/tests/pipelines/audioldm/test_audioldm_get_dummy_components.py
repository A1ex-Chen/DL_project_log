def get_dummy_components(self):
    torch.manual_seed(0)
    unet = UNet2DConditionModel(block_out_channels=(8, 16),
        layers_per_block=1, norm_num_groups=8, sample_size=32, in_channels=
        4, out_channels=4, down_block_types=('DownBlock2D',
        'CrossAttnDownBlock2D'), up_block_types=('CrossAttnUpBlock2D',
        'UpBlock2D'), cross_attention_dim=(8, 16), class_embed_type=
        'simple_projection', projection_class_embeddings_input_dim=8,
        class_embeddings_concat=True)
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=
        False)
    torch.manual_seed(0)
    vae = AutoencoderKL(block_out_channels=[8, 16], in_channels=1,
        out_channels=1, norm_num_groups=8, down_block_types=[
        'DownEncoderBlock2D', 'DownEncoderBlock2D'], up_block_types=[
        'UpDecoderBlock2D', 'UpDecoderBlock2D'], latent_channels=4)
    torch.manual_seed(0)
    text_encoder_config = ClapTextConfig(bos_token_id=0, eos_token_id=2,
        hidden_size=8, intermediate_size=37, layer_norm_eps=1e-05,
        num_attention_heads=1, num_hidden_layers=1, pad_token_id=1,
        vocab_size=1000, projection_dim=8)
    text_encoder = ClapTextModelWithProjection(text_encoder_config)
    tokenizer = RobertaTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-roberta', model_max_length=77)
    vocoder_config = SpeechT5HifiGanConfig(model_in_dim=8, sampling_rate=
        16000, upsample_initial_channel=16, upsample_rates=[2, 2],
        upsample_kernel_sizes=[4, 4], resblock_kernel_sizes=[3, 7],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]], normalize_before=False)
    vocoder = SpeechT5HifiGan(vocoder_config)
    components = {'unet': unet, 'scheduler': scheduler, 'vae': vae,
        'text_encoder': text_encoder, 'tokenizer': tokenizer, 'vocoder':
        vocoder}
    return components
