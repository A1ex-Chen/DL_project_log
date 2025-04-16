def setUp(self):
    self.model_tester = TFGPT2ModelTest.TFGPT2ModelTester(self)
    self.config_tester = ConfigTester(self, config_class=GPT2Config, n_embd=37)
