def create_and_check_xlm_sequence_classif(self, config, input_ids,
    token_type_ids, input_lengths, sequence_labels, token_labels,
    is_impossible_labels, input_mask):
    model = TFXLMForSequenceClassification(config)
    inputs = {'input_ids': input_ids, 'lengths': input_lengths}
    logits, = model(inputs)
    result = {'logits': logits.numpy()}
    self.parent.assertListEqual(list(result['logits'].shape), [self.
        batch_size, self.type_sequence_label_size])
