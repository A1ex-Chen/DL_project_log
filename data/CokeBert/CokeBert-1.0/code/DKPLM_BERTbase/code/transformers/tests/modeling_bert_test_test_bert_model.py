def test_bert_model(self, use_cuda=False):
    if use_cuda:
        self.model_tester.device = 'cuda'
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_bert_model(*config_and_inputs)
