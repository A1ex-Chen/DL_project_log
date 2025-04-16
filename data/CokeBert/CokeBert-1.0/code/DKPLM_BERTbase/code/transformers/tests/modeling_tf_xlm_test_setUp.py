def setUp(self):
    self.model_tester = TFXLMModelTest.TFXLMModelTester(self)
    self.config_tester = ConfigTester(self, config_class=XLMConfig, emb_dim=37)
