def get_dummy_components(self):
    torch.manual_seed(0)
    model = UNet2DConditionModel(act_fn='gelu', attention_head_dim=8,
        norm_num_groups=None, block_out_channels=[32, 32, 64, 64],
        time_cond_proj_dim=160, conv_in_kernel=1, conv_out_kernel=1,
        cross_attention_dim=32, down_block_types=('KDownBlock2D',
        'KCrossAttnDownBlock2D', 'KCrossAttnDownBlock2D',
        'KCrossAttnDownBlock2D'), in_channels=8, mid_block_type=None,
        only_cross_attention=False, out_channels=5, resnet_time_scale_shift
        ='scale_shift', time_embedding_type='fourier', timestep_post_act=
        'gelu', up_block_types=('KCrossAttnUpBlock2D',
        'KCrossAttnUpBlock2D', 'KCrossAttnUpBlock2D', 'KUpBlock2D'))
    vae = AutoencoderKL(block_out_channels=[32, 32, 64, 64], in_channels=3,
        out_channels=3, down_block_types=['DownEncoderBlock2D',
        'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D',
        'UpDecoderBlock2D', 'UpDecoderBlock2D'], latent_channels=4)
    scheduler = EulerDiscreteScheduler(prediction_type='sample')
    text_config = CLIPTextConfig(bos_token_id=0, eos_token_id=2,
        hidden_size=32, intermediate_size=37, layer_norm_eps=1e-05,
        num_attention_heads=4, num_hidden_layers=5, pad_token_id=1,
        vocab_size=1000, hidden_act='quick_gelu', projection_dim=512)
    text_encoder = CLIPTextModel(text_config)
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    components = {'unet': model.eval(), 'vae': vae.eval(), 'scheduler':
        scheduler, 'text_encoder': text_encoder, 'tokenizer': tokenizer}
    return components
