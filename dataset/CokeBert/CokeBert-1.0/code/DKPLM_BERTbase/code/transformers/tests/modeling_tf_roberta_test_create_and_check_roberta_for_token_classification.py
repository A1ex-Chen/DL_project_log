def create_and_check_roberta_for_token_classification(self, config,
    input_ids, token_type_ids, input_mask, sequence_labels, token_labels,
    choice_labels):
    config.num_labels = self.num_labels
    model = TFRobertaForTokenClassification(config=config)
    inputs = {'input_ids': input_ids, 'attention_mask': input_mask,
        'token_type_ids': token_type_ids}
    logits, = model(inputs)
    result = {'logits': logits.numpy()}
    self.parent.assertListEqual(list(result['logits'].shape), [self.
        batch_size, self.seq_length, self.num_labels])
