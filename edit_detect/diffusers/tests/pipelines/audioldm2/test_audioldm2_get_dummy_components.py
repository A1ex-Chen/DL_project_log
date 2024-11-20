def get_dummy_components(self):
    torch.manual_seed(0)
    unet = AudioLDM2UNet2DConditionModel(block_out_channels=(32, 64),
        layers_per_block=2, sample_size=32, in_channels=4, out_channels=4,
        down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'),
        cross_attention_dim=([None, 16, 32], [None, 16, 32]))
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=
        False)
    torch.manual_seed(0)
    vae = AutoencoderKL(block_out_channels=[32, 64], in_channels=1,
        out_channels=1, down_block_types=['DownEncoderBlock2D',
        'DownEncoderBlock2D'], up_block_types=['UpDecoderBlock2D',
        'UpDecoderBlock2D'], latent_channels=4)
    torch.manual_seed(0)
    text_branch_config = ClapTextConfig(bos_token_id=0, eos_token_id=2,
        hidden_size=16, intermediate_size=37, layer_norm_eps=1e-05,
        num_attention_heads=2, num_hidden_layers=2, pad_token_id=1,
        vocab_size=1000, projection_dim=16)
    audio_branch_config = ClapAudioConfig(spec_size=64, window_size=4,
        num_mel_bins=64, intermediate_size=37, layer_norm_eps=1e-05, depths
        =[2, 2], num_attention_heads=[2, 2], num_hidden_layers=2,
        hidden_size=192, projection_dim=16, patch_size=2, patch_stride=2,
        patch_embed_input_channels=4)
    text_encoder_config = ClapConfig.from_text_audio_configs(text_config=
        text_branch_config, audio_config=audio_branch_config, projection_dim=16
        )
    text_encoder = ClapModel(text_encoder_config)
    tokenizer = RobertaTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-roberta', model_max_length=77)
    feature_extractor = ClapFeatureExtractor.from_pretrained(
        'hf-internal-testing/tiny-random-ClapModel', hop_length=7900)
    torch.manual_seed(0)
    text_encoder_2_config = T5Config(vocab_size=32100, d_model=32, d_ff=37,
        d_kv=8, num_heads=2, num_layers=2)
    text_encoder_2 = T5EncoderModel(text_encoder_2_config)
    tokenizer_2 = T5Tokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-T5Model', model_max_length=77)
    torch.manual_seed(0)
    language_model_config = GPT2Config(n_embd=16, n_head=2, n_layer=2,
        vocab_size=1000, n_ctx=99, n_positions=99)
    language_model = GPT2Model(language_model_config)
    language_model.config.max_new_tokens = 8
    torch.manual_seed(0)
    projection_model = AudioLDM2ProjectionModel(text_encoder_dim=16,
        text_encoder_1_dim=32, langauge_model_dim=16)
    vocoder_config = SpeechT5HifiGanConfig(model_in_dim=8, sampling_rate=
        16000, upsample_initial_channel=16, upsample_rates=[2, 2],
        upsample_kernel_sizes=[4, 4], resblock_kernel_sizes=[3, 7],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]], normalize_before=False)
    vocoder = SpeechT5HifiGan(vocoder_config)
    components = {'unet': unet, 'scheduler': scheduler, 'vae': vae,
        'text_encoder': text_encoder, 'text_encoder_2': text_encoder_2,
        'tokenizer': tokenizer, 'tokenizer_2': tokenizer_2,
        'feature_extractor': feature_extractor, 'language_model':
        language_model, 'projection_model': projection_model, 'vocoder':
        vocoder}
    return components
