def create_and_check_xlm_lm_head(self, config, input_ids, token_type_ids,
    input_lengths, sequence_labels, token_labels, is_impossible_labels,
    input_mask):
    model = TFXLMWithLMHeadModel(config)
    inputs = {'input_ids': input_ids, 'lengths': input_lengths, 'langs':
        token_type_ids}
    outputs = model(inputs)
    logits = outputs[0]
    result = {'logits': logits.numpy()}
    self.parent.assertListEqual(list(result['logits'].shape), [self.
        batch_size, self.seq_length, self.vocab_size])
