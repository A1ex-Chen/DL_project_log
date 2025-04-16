def create_and_check_xlm_sequence_classif(self, config, input_ids,
    token_type_ids, input_lengths, sequence_labels, token_labels,
    is_impossible_labels, input_mask):
    model = XLMForSequenceClassification(config)
    model.eval()
    logits, = model(input_ids)
    loss, logits = model(input_ids, labels=sequence_labels)
    result = {'loss': loss, 'logits': logits}
    self.parent.assertListEqual(list(result['loss'].size()), [])
    self.parent.assertListEqual(list(result['logits'].size()), [self.
        batch_size, self.type_sequence_label_size])
