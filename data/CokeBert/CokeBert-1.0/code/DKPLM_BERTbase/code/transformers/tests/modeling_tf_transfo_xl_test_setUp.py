def setUp(self):
    self.model_tester = TFTransfoXLModelTest.TFTransfoXLModelTester(self)
    self.config_tester = ConfigTester(self, config_class=TransfoXLConfig,
        d_embed=37)
