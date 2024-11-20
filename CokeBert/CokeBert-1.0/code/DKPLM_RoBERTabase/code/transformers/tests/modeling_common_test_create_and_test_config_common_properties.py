def create_and_test_config_common_properties(self):
    config = self.config_class(**self.inputs_dict)
    self.parent.assertTrue(hasattr(config, 'vocab_size'))
    self.parent.assertTrue(hasattr(config, 'hidden_size'))
    self.parent.assertTrue(hasattr(config, 'num_attention_heads'))
    self.parent.assertTrue(hasattr(config, 'num_hidden_layers'))
