def prepare_config_and_inputs(self):
    input_ids_1 = ids_tensor([self.batch_size, self.seq_length], self.
        vocab_size)
    input_ids_2 = ids_tensor([self.batch_size, self.seq_length], self.
        vocab_size)
    lm_labels = None
    if self.use_labels:
        lm_labels = ids_tensor([self.batch_size, self.seq_length], self.
            vocab_size)
    config = TransfoXLConfig(vocab_size_or_config_json_file=self.vocab_size,
        mem_len=self.mem_len, clamp_len=self.clamp_len, cutoffs=self.
        cutoffs, d_model=self.hidden_size, d_embed=self.d_embed, n_head=
        self.num_attention_heads, d_head=self.d_head, d_inner=self.d_inner,
        div_val=self.div_val, n_layer=self.num_hidden_layers)
    return config, input_ids_1, input_ids_2, lm_labels
