def test_freeze_unet(self):

    def assert_frozen(module):
        for p in module.parameters():
            assert not p.requires_grad

    def assert_unfrozen(module):
        for p in module.parameters():
            assert p.requires_grad
    init_dict, _ = self.prepare_init_args_and_inputs_for_common()
    model = UNetControlNetXSModel(**init_dict)
    model.freeze_unet_params()
    modules_from_unet = [model.base_time_embedding, model.base_conv_in,
        model.base_conv_norm_out, model.base_conv_out]
    for m in modules_from_unet:
        assert_frozen(m)
    optional_modules_from_unet = [model.base_add_time_proj, model.
        base_add_embedding]
    for m in optional_modules_from_unet:
        if m is not None:
            assert_frozen(m)
    for i, d in enumerate(model.down_blocks):
        assert_frozen(d.base_resnets)
        if isinstance(d.base_attentions, nn.ModuleList):
            assert_frozen(d.base_attentions)
        if d.base_downsamplers is not None:
            assert_frozen(d.base_downsamplers)
    assert_frozen(model.mid_block.base_midblock)
    for i, u in enumerate(model.up_blocks):
        assert_frozen(u.resnets)
        if isinstance(u.attentions, nn.ModuleList):
            assert_frozen(u.attentions)
        if u.upsamplers is not None:
            assert_frozen(u.upsamplers)
    modules_from_controlnet = [model.controlnet_cond_embedding, model.
        ctrl_conv_in, model.control_to_base_for_conv_in]
    optional_modules_from_controlnet = [model.ctrl_time_embedding]
    for m in modules_from_controlnet:
        assert_unfrozen(m)
    for m in optional_modules_from_controlnet:
        if m is not None:
            assert_unfrozen(m)
    for d in model.down_blocks:
        assert_unfrozen(d.ctrl_resnets)
        assert_unfrozen(d.base_to_ctrl)
        assert_unfrozen(d.ctrl_to_base)
        if isinstance(d.ctrl_attentions, nn.ModuleList):
            assert_unfrozen(d.ctrl_attentions)
        if d.ctrl_downsamplers is not None:
            assert_unfrozen(d.ctrl_downsamplers)
    assert_unfrozen(model.mid_block.base_to_ctrl)
    assert_unfrozen(model.mid_block.ctrl_midblock)
    assert_unfrozen(model.mid_block.ctrl_to_base)
    for u in model.up_blocks:
        assert_unfrozen(u.ctrl_to_base)
