def test_resize_tokens_embeddings(self):
    original_config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    if not self.test_resize_embeddings:
        return
    for model_class in self.all_model_classes:
        config = copy.deepcopy(original_config)
        model = model_class(config)
        model_vocab_size = config.vocab_size
        model_embed = model.resize_token_embeddings(model_vocab_size)
        cloned_embeddings = model_embed.weight.clone()
        model_embed = model.resize_token_embeddings(model_vocab_size + 10)
        self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
        self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.
            shape[0] + 10)
        model_embed = model.resize_token_embeddings(model_vocab_size - 15)
        self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
        self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.
            shape[0] - 15)
        models_equal = True
        for p1, p2 in zip(cloned_embeddings, model_embed.weight):
            if p1.data.ne(p2.data).sum() > 0:
                models_equal = False
        self.assertTrue(models_equal)
