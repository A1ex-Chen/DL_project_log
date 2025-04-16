def prepare_config_and_inputs(self):
    input_ids_1 = ids_tensor([self.batch_size, self.seq_length], self.
        vocab_size)
    input_ids_2 = ids_tensor([self.batch_size, self.seq_length], self.
        vocab_size)
    segment_ids = ids_tensor([self.batch_size, self.seq_length], self.
        type_vocab_size)
    input_mask = ids_tensor([self.batch_size, self.seq_length], 2, dtype=tf
        .float32)
    input_ids_q = ids_tensor([self.batch_size, self.seq_length + 1], self.
        vocab_size)
    perm_mask = tf.zeros((self.batch_size, self.seq_length + 1, self.
        seq_length), dtype=tf.float32)
    perm_mask_last = tf.ones((self.batch_size, self.seq_length + 1, 1),
        dtype=tf.float32)
    perm_mask = tf.concat([perm_mask, perm_mask_last], axis=-1)
    target_mapping = tf.zeros((self.batch_size, 1, self.seq_length), dtype=
        tf.float32)
    target_mapping_last = tf.ones((self.batch_size, 1, 1), dtype=tf.float32)
    target_mapping = tf.concat([target_mapping, target_mapping_last], axis=-1)
    sequence_labels = None
    lm_labels = None
    is_impossible_labels = None
    if self.use_labels:
        lm_labels = ids_tensor([self.batch_size, self.seq_length], self.
            vocab_size)
        sequence_labels = ids_tensor([self.batch_size], self.
            type_sequence_label_size)
        is_impossible_labels = ids_tensor([self.batch_size], 2, dtype=tf.
            float32)
    config = XLNetConfig(vocab_size_or_config_json_file=self.vocab_size,
        d_model=self.hidden_size, n_head=self.num_attention_heads, d_inner=
        self.d_inner, n_layer=self.num_hidden_layers, untie_r=self.untie_r,
        max_position_embeddings=self.max_position_embeddings, mem_len=self.
        mem_len, clamp_len=self.clamp_len, same_length=self.same_length,
        reuse_len=self.reuse_len, bi_data=self.bi_data, initializer_range=
        self.initializer_range, num_labels=self.type_sequence_label_size)
    return (config, input_ids_1, input_ids_2, input_ids_q, perm_mask,
        input_mask, target_mapping, segment_ids, lm_labels, sequence_labels,
        is_impossible_labels)
