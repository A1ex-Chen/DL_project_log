def verify_param_count(orig_path, unet_diffusers_config):
    if '-II-' in orig_path:
        from deepfloyd_if.modules import IFStageII
        if_II = IFStageII(device='cpu', dir_or_name=orig_path)
    elif '-III-' in orig_path:
        from deepfloyd_if.modules import IFStageIII
        if_II = IFStageIII(device='cpu', dir_or_name=orig_path)
    else:
        assert f'Weird name. Should have -II- or -III- in path: {orig_path}'
    unet = UNet2DConditionModel(**unet_diffusers_config)
    assert_param_count(unet.time_embedding, if_II.model.time_embed)
    assert_param_count(unet.conv_in, if_II.model.input_blocks[:1])
    assert_param_count(unet.down_blocks[0], if_II.model.input_blocks[1:4])
    assert_param_count(unet.down_blocks[1], if_II.model.input_blocks[4:7])
    assert_param_count(unet.down_blocks[2], if_II.model.input_blocks[7:11])
    if '-II-' in orig_path:
        assert_param_count(unet.down_blocks[3], if_II.model.input_blocks[11:17]
            )
        assert_param_count(unet.down_blocks[4], if_II.model.input_blocks[17:])
    if '-III-' in orig_path:
        assert_param_count(unet.down_blocks[3], if_II.model.input_blocks[11:15]
            )
        assert_param_count(unet.down_blocks[4], if_II.model.input_blocks[15:20]
            )
        assert_param_count(unet.down_blocks[5], if_II.model.input_blocks[20:])
    assert_param_count(unet.mid_block, if_II.model.middle_block)
    if '-II-' in orig_path:
        assert_param_count(unet.up_blocks[0], if_II.model.output_blocks[:6])
        assert_param_count(unet.up_blocks[1], if_II.model.output_blocks[6:12])
        assert_param_count(unet.up_blocks[2], if_II.model.output_blocks[12:16])
        assert_param_count(unet.up_blocks[3], if_II.model.output_blocks[16:19])
        assert_param_count(unet.up_blocks[4], if_II.model.output_blocks[19:])
    if '-III-' in orig_path:
        assert_param_count(unet.up_blocks[0], if_II.model.output_blocks[:5])
        assert_param_count(unet.up_blocks[1], if_II.model.output_blocks[5:10])
        assert_param_count(unet.up_blocks[2], if_II.model.output_blocks[10:14])
        assert_param_count(unet.up_blocks[3], if_II.model.output_blocks[14:18])
        assert_param_count(unet.up_blocks[4], if_II.model.output_blocks[18:21])
        assert_param_count(unet.up_blocks[5], if_II.model.output_blocks[21:24])
    assert_param_count(unet.conv_norm_out, if_II.model.out[0])
    assert_param_count(unet.conv_out, if_II.model.out[2])
    assert_param_count(unet, if_II.model)
