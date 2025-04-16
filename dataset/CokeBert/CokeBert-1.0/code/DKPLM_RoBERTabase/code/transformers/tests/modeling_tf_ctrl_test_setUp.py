def setUp(self):
    self.model_tester = TFCTRLModelTest.TFCTRLModelTester(self)
    self.config_tester = ConfigTester(self, config_class=CTRLConfig, n_embd=37)
