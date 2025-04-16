def test_bert_model_as_decoder(self):
    config_and_inputs = (self.model_tester.
        prepare_config_and_inputs_for_decoder())
    self.model_tester.create_and_check_bert_model_as_decoder(*config_and_inputs
        )
