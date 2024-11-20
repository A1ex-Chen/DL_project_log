def create_and_check_presents(self, config, input_ids, token_type_ids,
    position_ids, mc_labels, lm_labels, mc_token_ids):
    for model_class in self.all_model_classes:
        model = model_class(config)
        model.eval()
        outputs = model(input_ids)
        presents = outputs[-1]
        self.parent.assertEqual(self.num_hidden_layers, len(presents))
        self.parent.assertListEqual(list(presents[0].size()), [2, self.
            batch_size * self.n_choices, self.num_attention_heads, self.
            seq_length, self.hidden_size // self.num_attention_heads])
