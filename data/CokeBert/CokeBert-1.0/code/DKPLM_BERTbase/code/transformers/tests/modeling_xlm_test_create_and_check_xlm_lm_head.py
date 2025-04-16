def create_and_check_xlm_lm_head(self, config, input_ids, token_type_ids,
    input_lengths, sequence_labels, token_labels, is_impossible_labels,
    input_mask):
    model = XLMWithLMHeadModel(config)
    model.eval()
    loss, logits = model(input_ids, token_type_ids=token_type_ids, labels=
        token_labels)
    result = {'loss': loss, 'logits': logits}
    self.parent.assertListEqual(list(result['loss'].size()), [])
    self.parent.assertListEqual(list(result['logits'].size()), [self.
        batch_size, self.seq_length, self.vocab_size])
