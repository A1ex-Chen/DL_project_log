def setUp(self):
    self.model_tester = BertModelTest.BertModelTester(self)
    self.config_tester = ConfigTester(self, config_class=BertConfig,
        hidden_size=37)
