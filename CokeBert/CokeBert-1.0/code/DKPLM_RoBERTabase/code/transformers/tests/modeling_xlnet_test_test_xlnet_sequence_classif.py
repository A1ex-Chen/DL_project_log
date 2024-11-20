def test_xlnet_sequence_classif(self):
    self.model_tester.set_seed()
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_xlnet_sequence_classif(*
        config_and_inputs)
