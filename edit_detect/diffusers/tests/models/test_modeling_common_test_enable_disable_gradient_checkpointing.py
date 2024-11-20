@require_torch_accelerator_with_training
def test_enable_disable_gradient_checkpointing(self):
    if not self.model_class._supports_gradient_checkpointing:
        return
    init_dict, _ = self.prepare_init_args_and_inputs_for_common()
    model = self.model_class(**init_dict)
    self.assertFalse(model.is_gradient_checkpointing)
    model.enable_gradient_checkpointing()
    self.assertTrue(model.is_gradient_checkpointing)
    model.disable_gradient_checkpointing()
    self.assertFalse(model.is_gradient_checkpointing)
