def setUp(self):
    self.model_tester = TFOpenAIGPTModelTest.TFOpenAIGPTModelTester(self)
    self.config_tester = ConfigTester(self, config_class=OpenAIGPTConfig,
        n_embd=37)
