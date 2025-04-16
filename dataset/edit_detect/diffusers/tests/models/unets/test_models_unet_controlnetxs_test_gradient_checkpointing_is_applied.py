def test_gradient_checkpointing_is_applied(self):
    model_class_copy = copy.copy(UNetControlNetXSModel)
    modules_with_gc_enabled = {}

    def _set_gradient_checkpointing_new(self, module, value=False):
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = value
            modules_with_gc_enabled[module.__class__.__name__] = True
    model_class_copy._set_gradient_checkpointing = (
        _set_gradient_checkpointing_new)
    init_dict, _ = self.prepare_init_args_and_inputs_for_common()
    model = model_class_copy(**init_dict)
    model.enable_gradient_checkpointing()
    EXPECTED_SET = {'Transformer2DModel', 'UNetMidBlock2DCrossAttn',
        'ControlNetXSCrossAttnDownBlock2D',
        'ControlNetXSCrossAttnMidBlock2D', 'ControlNetXSCrossAttnUpBlock2D'}
    assert set(modules_with_gc_enabled.keys()) == EXPECTED_SET
    assert all(modules_with_gc_enabled.values()
        ), 'All modules should be enabled'
