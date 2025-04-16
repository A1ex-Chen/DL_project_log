def test_conditioning_channels(self):
    unet = UNet2DConditionModel(block_out_channels=(32, 64),
        layers_per_block=2, sample_size=32, in_channels=4, out_channels=4,
        down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'), mid_block_type=
        'UNetMidBlock2D', attention_head_dim=(2, 4), use_linear_projection=
        True, addition_embed_type='text_time', addition_time_embed_dim=8,
        transformer_layers_per_block=(1, 2),
        projection_class_embeddings_input_dim=80, cross_attention_dim=64,
        time_cond_proj_dim=None)
    controlnet = ControlNetModel.from_unet(unet, conditioning_channels=4)
    assert type(controlnet.mid_block) == UNetMidBlock2D
    assert controlnet.conditioning_channels == 4
