def create_and_check_distilbert_for_sequence_classification(self, config,
    input_ids, input_mask, sequence_labels, token_labels, choice_labels):
    config.num_labels = self.num_labels
    model = TFDistilBertForSequenceClassification(config)
    inputs = {'input_ids': input_ids, 'attention_mask': input_mask}
    logits, = model(inputs)
    result = {'logits': logits.numpy()}
    self.parent.assertListEqual(list(result['logits'].shape), [self.
        batch_size, self.num_labels])
