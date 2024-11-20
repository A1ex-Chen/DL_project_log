def setUp(self):
    self.model_tester = DistilBertModelTest.DistilBertModelTester(self)
    self.config_tester = ConfigTester(self, config_class=DistilBertConfig,
        dim=37)
