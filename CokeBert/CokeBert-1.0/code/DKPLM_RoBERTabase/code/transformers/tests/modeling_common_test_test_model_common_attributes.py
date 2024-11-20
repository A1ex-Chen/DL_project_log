def test_model_common_attributes(self):
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    for model_class in self.all_model_classes:
        model = model_class(config)
        self.assertIsInstance(model.get_input_embeddings(), (torch.nn.
            Embedding, AdaptiveEmbedding))
        model.set_input_embeddings(torch.nn.Embedding(10, 10))
        x = model.get_output_embeddings()
        self.assertTrue(x is None or isinstance(x, torch.nn.Linear))
