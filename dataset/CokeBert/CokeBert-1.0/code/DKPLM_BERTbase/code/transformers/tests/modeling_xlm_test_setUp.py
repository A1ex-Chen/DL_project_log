def setUp(self):
    self.model_tester = XLMModelTest.XLMModelTester(self)
    self.config_tester = ConfigTester(self, config_class=XLMConfig, emb_dim=37)
