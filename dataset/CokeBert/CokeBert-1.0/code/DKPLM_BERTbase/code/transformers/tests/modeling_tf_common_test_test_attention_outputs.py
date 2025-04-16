def test_attention_outputs(self):
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    for model_class in self.all_model_classes:
        config.output_attentions = True
        config.output_hidden_states = False
        model = model_class(config)
        outputs = model(inputs_dict)
        attentions = [t.numpy() for t in outputs[-1]]
        self.assertEqual(model.config.output_attentions, True)
        self.assertEqual(model.config.output_hidden_states, False)
        self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
        self.assertListEqual(list(attentions[0].shape[-3:]), [self.
            model_tester.num_attention_heads, self.model_tester.seq_length,
            self.model_tester.key_len if hasattr(self.model_tester,
            'key_len') else self.model_tester.seq_length])
        out_len = len(outputs)
        config.output_attentions = True
        config.output_hidden_states = True
        model = model_class(config)
        outputs = model(inputs_dict)
        self.assertEqual(out_len + 1, len(outputs))
        self.assertEqual(model.config.output_attentions, True)
        self.assertEqual(model.config.output_hidden_states, True)
        attentions = [t.numpy() for t in outputs[-1]]
        self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
        self.assertListEqual(list(attentions[0].shape[-3:]), [self.
            model_tester.num_attention_heads, self.model_tester.seq_length,
            self.model_tester.key_len if hasattr(self.model_tester,
            'key_len') else self.model_tester.seq_length])
