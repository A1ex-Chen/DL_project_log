def test_tie_model_weights(self):
    if not self.test_torchscript:
        return
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())

    def check_same_values(layer_1, layer_2):
        equal = True
        for p1, p2 in zip(layer_1.weight, layer_2.weight):
            if p1.data.ne(p2.data).sum() > 0:
                equal = False
        return equal
    for model_class in self.all_model_classes:
        config.torchscript = True
        model_not_tied = model_class(config)
        if model_not_tied.get_output_embeddings() is None:
            continue
        params_not_tied = list(model_not_tied.parameters())
        config_tied = copy.deepcopy(config)
        config_tied.torchscript = False
        model_tied = model_class(config_tied)
        params_tied = list(model_tied.parameters())
        self.assertGreater(len(params_not_tied), len(params_tied))
        model_tied.resize_token_embeddings(config.vocab_size + 10)
        params_tied_2 = list(model_tied.parameters())
        self.assertGreater(len(params_not_tied), len(params_tied))
        self.assertEqual(len(params_tied_2), len(params_tied))
