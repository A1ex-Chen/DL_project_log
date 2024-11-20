def test_torchscript_output_attentions(self):
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    config.output_attentions = True
    self._create_and_check_torchscript(config, inputs_dict)
