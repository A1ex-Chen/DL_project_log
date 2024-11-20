def setUp(self):
    self.model_tester = TFXLNetModelTest.TFXLNetModelTester(self)
    self.config_tester = ConfigTester(self, config_class=XLNetConfig,
        d_inner=37)
