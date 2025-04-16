def prepare_config_and_inputs_for_decoder(self):
    (config, input_ids, token_type_ids, input_mask, sequence_labels,
        token_labels, choice_labels) = self.prepare_config_and_inputs()
    config.is_decoder = True
    encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length,
        self.hidden_size])
    encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length],
        vocab_size=2)
    return (config, input_ids, token_type_ids, input_mask, sequence_labels,
        token_labels, choice_labels, encoder_hidden_states,
        encoder_attention_mask)
