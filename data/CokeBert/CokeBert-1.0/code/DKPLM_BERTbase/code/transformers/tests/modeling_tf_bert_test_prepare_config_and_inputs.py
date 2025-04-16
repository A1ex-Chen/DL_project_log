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
    sequence_labels = None
    token_labels = None
    choice_labels = None
    if self.use_labels:
        sequence_labels = ids_tensor([self.batch_size], self.
            type_sequence_label_size)
        token_labels = ids_tensor([self.batch_size, self.seq_length], self.
            num_labels)
        choice_labels = ids_tensor([self.batch_size], self.num_choices)
    config = BertConfig(vocab_size_or_config_json_file=self.vocab_size,
        hidden_size=self.hidden_size, num_hidden_layers=self.
        num_hidden_layers, num_attention_heads=self.num_attention_heads,
        intermediate_size=self.intermediate_size, hidden_act=self.
        hidden_act, hidden_dropout_prob=self.hidden_dropout_prob,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        max_position_embeddings=self.max_position_embeddings,
        type_vocab_size=self.type_vocab_size, initializer_range=self.
        initializer_range)
    return (config, input_ids, token_type_ids, input_mask, sequence_labels,
        token_labels, choice_labels)
