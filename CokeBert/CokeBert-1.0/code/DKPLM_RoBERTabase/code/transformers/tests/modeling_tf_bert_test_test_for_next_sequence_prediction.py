def test_for_next_sequence_prediction(self):
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_bert_for_next_sequence_prediction(*
        config_and_inputs)
