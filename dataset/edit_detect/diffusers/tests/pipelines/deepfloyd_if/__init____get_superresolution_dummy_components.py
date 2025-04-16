def _get_superresolution_dummy_components(self):
    torch.manual_seed(0)
    text_encoder = T5EncoderModel.from_pretrained(
        'hf-internal-testing/tiny-random-t5')
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-t5')
    torch.manual_seed(0)
    unet = UNet2DConditionModel(sample_size=32, layers_per_block=[1, 2],
        block_out_channels=[32, 64], down_block_types=[
        'ResnetDownsampleBlock2D', 'SimpleCrossAttnDownBlock2D'],
        mid_block_type='UNetMidBlock2DSimpleCrossAttn', up_block_types=[
        'SimpleCrossAttnUpBlock2D', 'ResnetUpsampleBlock2D'], in_channels=6,
        out_channels=6, cross_attention_dim=32, encoder_hid_dim=32,
        attention_head_dim=8, addition_embed_type='text',
        addition_embed_type_num_heads=2, cross_attention_norm='group_norm',
        resnet_time_scale_shift='scale_shift', act_fn='gelu',
        class_embed_type='timestep', mid_block_scale_factor=1.414,
        time_embedding_act_fn='gelu', time_embedding_dim=32)
    unet.set_attn_processor(AttnAddedKVProcessor())
    torch.manual_seed(0)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule=
        'squaredcos_cap_v2', beta_start=0.0001, beta_end=0.02, thresholding
        =True, dynamic_thresholding_ratio=0.95, sample_max_value=1.0,
        prediction_type='epsilon', variance_type='learned_range')
    torch.manual_seed(0)
    image_noising_scheduler = DDPMScheduler(num_train_timesteps=1000,
        beta_schedule='squaredcos_cap_v2', beta_start=0.0001, beta_end=0.02)
    torch.manual_seed(0)
    watermarker = IFWatermarker()
    return {'text_encoder': text_encoder, 'tokenizer': tokenizer, 'unet':
        unet, 'scheduler': scheduler, 'image_noising_scheduler':
        image_noising_scheduler, 'watermarker': watermarker,
        'safety_checker': None, 'feature_extractor': None}
