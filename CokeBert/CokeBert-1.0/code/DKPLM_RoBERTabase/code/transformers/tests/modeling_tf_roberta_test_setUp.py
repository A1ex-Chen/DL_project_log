def setUp(self):
    self.model_tester = TFRobertaModelTest.TFRobertaModelTester(self)
    self.config_tester = ConfigTester(self, config_class=RobertaConfig,
        hidden_size=37)
