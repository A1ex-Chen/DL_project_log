def prepare_config_and_inputs(self):
    input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
    input_mask = ids_tensor([self.batch_size, self.seq_length], 2).float()
    input_lengths = None
    if self.use_input_lengths:
        input_lengths = ids_tensor([self.batch_size], vocab_size=2
            ) + self.seq_length - 2
    token_type_ids = None
    if self.use_token_type_ids:
        token_type_ids = ids_tensor([self.batch_size, self.seq_length],
            self.n_langs)
    sequence_labels = None
    token_labels = None
    is_impossible_labels = None
    if self.use_labels:
        sequence_labels = ids_tensor([self.batch_size], self.
            type_sequence_label_size)
        token_labels = ids_tensor([self.batch_size, self.seq_length], self.
            num_labels)
        is_impossible_labels = ids_tensor([self.batch_size], 2).float()
    config = XLMConfig(vocab_size_or_config_json_file=self.vocab_size,
        n_special=self.n_special, emb_dim=self.hidden_size, n_layers=self.
        num_hidden_layers, n_heads=self.num_attention_heads, dropout=self.
        hidden_dropout_prob, attention_dropout=self.
        attention_probs_dropout_prob, gelu_activation=self.gelu_activation,
        sinusoidal_embeddings=self.sinusoidal_embeddings, asm=self.asm,
        causal=self.causal, n_langs=self.n_langs, max_position_embeddings=
        self.max_position_embeddings, initializer_range=self.
        initializer_range, summary_type=self.summary_type, use_proj=self.
        use_proj)
    return (config, input_ids, token_type_ids, input_lengths,
        sequence_labels, token_labels, is_impossible_labels, input_mask)
