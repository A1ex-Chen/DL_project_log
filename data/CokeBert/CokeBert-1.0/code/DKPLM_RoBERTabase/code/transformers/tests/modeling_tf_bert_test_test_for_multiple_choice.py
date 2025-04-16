def test_for_multiple_choice(self):
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_bert_for_multiple_choice(*
        config_and_inputs)
