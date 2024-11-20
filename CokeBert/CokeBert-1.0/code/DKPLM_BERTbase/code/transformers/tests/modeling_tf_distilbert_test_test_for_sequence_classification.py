def test_for_sequence_classification(self):
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_distilbert_for_sequence_classification(*
        config_and_inputs)
