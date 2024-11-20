def test_hidden_states_output(self):
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    for model_class in self.all_model_classes:
        config.output_hidden_states = True
        config.output_attentions = False
        model = model_class(config)
        outputs = model(inputs_dict)
        hidden_states = [t.numpy() for t in outputs[-1]]
        self.assertEqual(model.config.output_attentions, False)
        self.assertEqual(model.config.output_hidden_states, True)
        self.assertEqual(len(hidden_states), self.model_tester.
            num_hidden_layers + 1)
        self.assertListEqual(list(hidden_states[0].shape[-2:]), [self.
            model_tester.seq_length, self.model_tester.hidden_size])
