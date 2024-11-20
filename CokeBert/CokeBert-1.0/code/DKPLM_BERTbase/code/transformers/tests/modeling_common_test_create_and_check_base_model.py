def create_and_check_base_model(self, config, input_ids, token_type_ids,
    position_ids, mc_labels, lm_labels, mc_token_ids):
    model = self.base_model_class(config)
    model.eval()
    outputs = model(input_ids, position_ids, token_type_ids)
    outputs = model(input_ids, position_ids)
    outputs = model(input_ids)
    hidden_state = outputs[0]
    self.parent.assertListEqual(list(hidden_state.size()), [self.batch_size,
        self.n_choices, self.seq_length, self.hidden_size])
