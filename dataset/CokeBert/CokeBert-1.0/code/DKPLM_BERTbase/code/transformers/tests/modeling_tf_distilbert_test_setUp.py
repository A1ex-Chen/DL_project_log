def setUp(self):
    self.model_tester = TFDistilBertModelTest.TFDistilBertModelTester(self)
    self.config_tester = ConfigTester(self, config_class=DistilBertConfig,
        dim=37)
