def get_pipeline_components(self):
    unet = UNet2DConditionModel(block_out_channels=(32, 64),
        layers_per_block=2, sample_size=32, in_channels=4, out_channels=4,
        down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'),
        cross_attention_dim=32)
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=
        False)
    vae = AutoencoderKL(block_out_channels=[32, 64], in_channels=3,
        out_channels=3, down_block_types=['DownEncoderBlock2D',
        'DownEncoderBlock2D'], up_block_types=['UpDecoderBlock2D',
        'UpDecoderBlock2D'], latent_channels=4)
    text_encoder_config = CLIPTextConfig(bos_token_id=0, eos_token_id=2,
        hidden_size=32, intermediate_size=37, layer_norm_eps=1e-05,
        num_attention_heads=4, num_hidden_layers=5, pad_token_id=1,
        vocab_size=1000)
    text_encoder = CLIPTextModel(text_encoder_config)
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_vocab = {'<|startoftext|>': 0, '<|endoftext|>': 1, '!': 2}
        vocab_path = os.path.join(tmpdir, 'vocab.json')
        with open(vocab_path, 'w') as f:
            json.dump(dummy_vocab, f)
        merges = 'Ġ t\nĠt h'
        merges_path = os.path.join(tmpdir, 'merges.txt')
        with open(merges_path, 'w') as f:
            f.writelines(merges)
        tokenizer = CLIPTokenizer(vocab_file=vocab_path, merges_file=
            merges_path)
    components = {'unet': unet, 'scheduler': scheduler, 'vae': vae,
        'text_encoder': text_encoder, 'tokenizer': tokenizer,
        'safety_checker': None, 'feature_extractor': None}
    return components
