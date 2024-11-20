def get_dummy_components(self):
    torch.manual_seed(0)
    unet = UNet2DConditionModel(block_out_channels=(32, 64),
        layers_per_block=2, sample_size=32, in_channels=5, out_channels=4,
        down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'),
        cross_attention_dim=32, attention_head_dim=(2, 4),
        use_linear_projection=True)
    scheduler = PNDMScheduler(skip_prk_steps=True)
    torch.manual_seed(0)
    vae = AutoencoderKL(block_out_channels=[32, 64], in_channels=3,
        out_channels=3, down_block_types=['DownEncoderBlock2D',
        'DownEncoderBlock2D'], up_block_types=['UpDecoderBlock2D',
        'UpDecoderBlock2D'], latent_channels=4)
    torch.manual_seed(0)
    text_encoder_config = CLIPTextConfig(bos_token_id=0, eos_token_id=2,
        hidden_size=32, intermediate_size=37, layer_norm_eps=1e-05,
        num_attention_heads=4, num_hidden_layers=5, pad_token_id=1,
        vocab_size=1000)
    text_encoder = CLIPTextModel(text_encoder_config)
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    backbone_config = {'global_padding': 'same', 'layer_type': 'bottleneck',
        'depths': [3, 4, 9], 'out_features': ['stage1', 'stage2', 'stage3'],
        'embedding_dynamic_padding': True, 'hidden_sizes': [96, 192, 384, 
        768], 'num_groups': 2}
    depth_estimator_config = DPTConfig(image_size=32, patch_size=16,
        num_channels=3, hidden_size=32, num_hidden_layers=4,
        backbone_out_indices=(0, 1, 2, 3), num_attention_heads=4,
        intermediate_size=37, hidden_act='gelu', hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1, is_decoder=False,
        initializer_range=0.02, is_hybrid=True, backbone_config=
        backbone_config, backbone_featmap_shape=[1, 384, 24, 24])
    depth_estimator = DPTForDepthEstimation(depth_estimator_config).eval()
    feature_extractor = DPTFeatureExtractor.from_pretrained(
        'hf-internal-testing/tiny-random-DPTForDepthEstimation')
    components = {'unet': unet, 'scheduler': scheduler, 'vae': vae,
        'text_encoder': text_encoder, 'tokenizer': tokenizer,
        'depth_estimator': depth_estimator, 'feature_extractor':
        feature_extractor}
    return components
