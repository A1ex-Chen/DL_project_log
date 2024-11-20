def prepare_config_and_inputs(self):
    total_num_tokens = self.vocab_size
    input_ids = ids_tensor([self.batch_size, self.n_choices, self.
        seq_length], total_num_tokens)
    position_ids = None
    if self.use_position_ids:
        position_ids = ids_tensor([self.batch_size, self.n_choices, self.
            seq_length], self.n_positions)
    token_type_ids = None
    if self.use_token_type_ids:
        total_voc = self.vocab_size
        token_type_ids = ids_tensor([self.batch_size, self.n_choices, self.
            seq_length], total_voc)
    mc_labels = None
    lm_labels = None
    mc_token_ids = None
    if self.use_labels:
        mc_labels = ids_tensor([self.batch_size], self.type_sequence_label_size
            )
        lm_labels = ids_tensor([self.batch_size, self.n_choices, self.
            seq_length], self.num_labels)
        mc_token_ids = ids_tensor([self.batch_size, self.n_choices], self.
            seq_length)
    config = self.config_class(vocab_size_or_config_json_file=self.
        vocab_size, n_positions=self.n_positions, n_embd=self.hidden_size,
        n_layer=self.num_hidden_layers, n_head=self.num_attention_heads,
        initializer_range=self.initializer_range)
    return (config, input_ids, token_type_ids, position_ids, mc_labels,
        lm_labels, mc_token_ids)
