def test_inputs_embeds(self):
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    input_ids = inputs_dict['input_ids']
    del inputs_dict['input_ids']
    for model_class in self.all_model_classes:
        model = model_class(config)
        wte = model.get_input_embeddings()
        try:
            x = wte(input_ids, mode='embedding')
        except:
            try:
                x = wte([input_ids], mode='embedding')
            except:
                x = tf.ones(input_ids.shape + [self.model_tester.
                    hidden_size], dtype=tf.dtypes.float32)
        inputs_dict['inputs_embeds'] = x
        outputs = model(inputs_dict)
