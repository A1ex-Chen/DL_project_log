def test_from_unet(self):
    unet = self.get_dummy_unet()
    controlnet = self.get_dummy_controlnet_from_unet(unet)
    model = UNetControlNetXSModel.from_unet(unet, controlnet)
    model_state_dict = model.state_dict()

    def assert_equal_weights(module, weight_dict_prefix):
        for param_name, param_value in module.named_parameters():
            assert torch.equal(model_state_dict[weight_dict_prefix + '.' +
                param_name], param_value)
    modules_from_unet = ['time_embedding', 'conv_in', 'conv_norm_out',
        'conv_out']
    for p in modules_from_unet:
        assert_equal_weights(getattr(unet, p), 'base_' + p)
    optional_modules_from_unet = ['class_embedding', 'add_time_proj',
        'add_embedding']
    for p in optional_modules_from_unet:
        if hasattr(unet, p) and getattr(unet, p) is not None:
            assert_equal_weights(getattr(unet, p), 'base_' + p)
    assert len(unet.down_blocks) == len(model.down_blocks)
    for i, d in enumerate(unet.down_blocks):
        assert_equal_weights(d.resnets, f'down_blocks.{i}.base_resnets')
        if hasattr(d, 'attentions'):
            assert_equal_weights(d.attentions,
                f'down_blocks.{i}.base_attentions')
        if hasattr(d, 'downsamplers') and getattr(d, 'downsamplers'
            ) is not None:
            assert_equal_weights(d.downsamplers[0],
                f'down_blocks.{i}.base_downsamplers')
    assert_equal_weights(unet.mid_block, 'mid_block.base_midblock')
    assert len(unet.up_blocks) == len(model.up_blocks)
    for i, u in enumerate(unet.up_blocks):
        assert_equal_weights(u.resnets, f'up_blocks.{i}.resnets')
        if hasattr(u, 'attentions'):
            assert_equal_weights(u.attentions, f'up_blocks.{i}.attentions')
        if hasattr(u, 'upsamplers') and getattr(u, 'upsamplers') is not None:
            assert_equal_weights(u.upsamplers[0], f'up_blocks.{i}.upsamplers')
    modules_from_controlnet = {'controlnet_cond_embedding':
        'controlnet_cond_embedding', 'conv_in': 'ctrl_conv_in',
        'control_to_base_for_conv_in': 'control_to_base_for_conv_in'}
    optional_modules_from_controlnet = {'time_embedding': 'ctrl_time_embedding'
        }
    for name_in_controlnet, name_in_unetcnxs in modules_from_controlnet.items(
        ):
        assert_equal_weights(getattr(controlnet, name_in_controlnet),
            name_in_unetcnxs)
    for name_in_controlnet, name_in_unetcnxs in optional_modules_from_controlnet.items(
        ):
        if hasattr(controlnet, name_in_controlnet) and getattr(controlnet,
            name_in_controlnet) is not None:
            assert_equal_weights(getattr(controlnet, name_in_controlnet),
                name_in_unetcnxs)
    assert len(controlnet.down_blocks) == len(model.down_blocks)
    for i, d in enumerate(controlnet.down_blocks):
        assert_equal_weights(d.resnets, f'down_blocks.{i}.ctrl_resnets')
        assert_equal_weights(d.base_to_ctrl, f'down_blocks.{i}.base_to_ctrl')
        assert_equal_weights(d.ctrl_to_base, f'down_blocks.{i}.ctrl_to_base')
        if d.attentions is not None:
            assert_equal_weights(d.attentions,
                f'down_blocks.{i}.ctrl_attentions')
        if d.downsamplers is not None:
            assert_equal_weights(d.downsamplers,
                f'down_blocks.{i}.ctrl_downsamplers')
    assert_equal_weights(controlnet.mid_block.base_to_ctrl,
        'mid_block.base_to_ctrl')
    assert_equal_weights(controlnet.mid_block.midblock,
        'mid_block.ctrl_midblock')
    assert_equal_weights(controlnet.mid_block.ctrl_to_base,
        'mid_block.ctrl_to_base')
    assert len(controlnet.up_connections) == len(model.up_blocks)
    for i, u in enumerate(controlnet.up_connections):
        assert_equal_weights(u.ctrl_to_base, f'up_blocks.{i}.ctrl_to_base')
