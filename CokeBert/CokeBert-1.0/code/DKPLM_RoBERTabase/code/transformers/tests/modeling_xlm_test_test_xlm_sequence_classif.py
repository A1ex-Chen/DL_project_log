def test_xlm_sequence_classif(self):
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_xlm_sequence_classif(*config_and_inputs)
