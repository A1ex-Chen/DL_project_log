def test_freeze_unet2d(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    model = self.model_class(**init_dict)
    model.freeze_unet2d_params()
    for param_name, param_value in model.named_parameters():
        if 'motion_modules' not in param_name:
            self.assertFalse(param_value.requires_grad)
        else:
            self.assertTrue(param_value.requires_grad)
