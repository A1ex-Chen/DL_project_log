def get_dummy_components(self):
    torch.manual_seed(0)
    unet = UNet2DConditionModel(block_out_channels=(4, 8), layers_per_block
        =2, sample_size=32, in_channels=4, out_channels=4, down_block_types
        =('DownBlock2D', 'CrossAttnDownBlock2D'), up_block_types=(
        'CrossAttnUpBlock2D', 'UpBlock2D'), cross_attention_dim=32,
        norm_num_groups=1)
    torch.manual_seed(0)

    def init_weights(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.normal_(m.weight)
            m.bias.data.fill_(1.0)
    controlnet1 = ControlNetModel(block_out_channels=(4, 8),
        layers_per_block=2, in_channels=4, down_block_types=('DownBlock2D',
        'CrossAttnDownBlock2D'), cross_attention_dim=32,
        conditioning_embedding_out_channels=(16, 32), norm_num_groups=1)
    controlnet1.controlnet_down_blocks.apply(init_weights)
    torch.manual_seed(0)
    controlnet2 = ControlNetModel(block_out_channels=(4, 8),
        layers_per_block=2, in_channels=4, down_block_types=('DownBlock2D',
        'CrossAttnDownBlock2D'), cross_attention_dim=32,
        conditioning_embedding_out_channels=(16, 32), norm_num_groups=1)
    controlnet2.controlnet_down_blocks.apply(init_weights)
    torch.manual_seed(0)
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=
        False)
    torch.manual_seed(0)
    vae = AutoencoderKL(block_out_channels=[4, 8], in_channels=3,
        out_channels=3, down_block_types=['DownEncoderBlock2D',
        'DownEncoderBlock2D'], up_block_types=['UpDecoderBlock2D',
        'UpDecoderBlock2D'], latent_channels=4, norm_num_groups=2)
    torch.manual_seed(0)
    text_encoder_config = CLIPTextConfig(bos_token_id=0, eos_token_id=2,
        hidden_size=32, intermediate_size=37, layer_norm_eps=1e-05,
        num_attention_heads=4, num_hidden_layers=5, pad_token_id=1,
        vocab_size=1000)
    text_encoder = CLIPTextModel(text_encoder_config)
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    controlnet = MultiControlNetModel([controlnet1, controlnet2])
    components = {'unet': unet, 'controlnet': controlnet, 'scheduler':
        scheduler, 'vae': vae, 'text_encoder': text_encoder, 'tokenizer':
        tokenizer, 'safety_checker': None, 'feature_extractor': None,
        'image_encoder': None}
    return components
