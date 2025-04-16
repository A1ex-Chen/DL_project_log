def test_gradient_checkpointing_is_applied(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    model_class_copy = copy.copy(self.model_class)
    modules_with_gc_enabled = {}

    def _set_gradient_checkpointing_new(self, module, value=False):
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = value
            modules_with_gc_enabled[module.__class__.__name__] = True
    model_class_copy._set_gradient_checkpointing = (
        _set_gradient_checkpointing_new)
    model = model_class_copy(**init_dict)
    model.enable_gradient_checkpointing()
    EXPECTED_SET = {'CrossAttnUpBlockMotion', 'CrossAttnDownBlockMotion',
        'UNetMidBlockCrossAttnMotion', 'UpBlockMotion',
        'Transformer2DModel', 'DownBlockMotion'}
    assert set(modules_with_gc_enabled.keys()) == EXPECTED_SET
    assert all(modules_with_gc_enabled.values()
        ), 'All modules should be enabled'
