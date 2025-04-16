def test_transfo_xl_model(self):
    self.model_tester.set_seed()
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    output_result = self.model_tester.create_transfo_xl_model(*
        config_and_inputs)
    self.model_tester.check_transfo_xl_model_output(output_result)
