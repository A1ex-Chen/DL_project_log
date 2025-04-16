def test_for_token_classification(self):
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_bert_for_token_classification(*
        config_and_inputs)
