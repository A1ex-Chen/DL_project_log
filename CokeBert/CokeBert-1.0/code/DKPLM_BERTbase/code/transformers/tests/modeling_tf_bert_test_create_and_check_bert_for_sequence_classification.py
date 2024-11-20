def create_and_check_bert_for_sequence_classification(self, config,
    input_ids, token_type_ids, input_mask, sequence_labels, token_labels,
    choice_labels):
    config.num_labels = self.num_labels
    model = TFBertForSequenceClassification(config=config)
    inputs = {'input_ids': input_ids, 'attention_mask': input_mask,
        'token_type_ids': token_type_ids}
    logits, = model(inputs)
    result = {'logits': logits.numpy()}
    self.parent.assertListEqual(list(result['logits'].shape), [self.
        batch_size, self.num_labels])
