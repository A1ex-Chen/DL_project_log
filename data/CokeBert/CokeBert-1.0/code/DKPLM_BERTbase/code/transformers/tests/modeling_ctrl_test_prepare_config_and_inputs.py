def prepare_config_and_inputs(self):
    input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
    input_mask = None
    if self.use_input_mask:
        input_mask = ids_tensor([self.batch_size, self.seq_length],
            vocab_size=2)
    token_type_ids = None
    if self.use_token_type_ids:
        token_type_ids = ids_tensor([self.batch_size, self.seq_length],
            self.type_vocab_size)
    mc_token_ids = None
    if self.use_mc_token_ids:
        mc_token_ids = ids_tensor([self.batch_size, self.num_choices], self
            .seq_length)
    sequence_labels = None
    token_labels = None
    choice_labels = None
    if self.use_labels:
        sequence_labels = ids_tensor([self.batch_size], self.
            type_sequence_label_size)
        token_labels = ids_tensor([self.batch_size, self.seq_length], self.
            num_labels)
        choice_labels = ids_tensor([self.batch_size], self.num_choices)
    config = CTRLConfig(vocab_size_or_config_json_file=self.vocab_size,
        n_embd=self.hidden_size, n_layer=self.num_hidden_layers, n_head=
        self.num_attention_heads, n_positions=self.max_position_embeddings,
        n_ctx=self.max_position_embeddings)
    head_mask = ids_tensor([self.num_hidden_layers, self.
        num_attention_heads], 2)
    return (config, input_ids, input_mask, head_mask, token_type_ids,
        mc_token_ids, sequence_labels, token_labels, choice_labels)
