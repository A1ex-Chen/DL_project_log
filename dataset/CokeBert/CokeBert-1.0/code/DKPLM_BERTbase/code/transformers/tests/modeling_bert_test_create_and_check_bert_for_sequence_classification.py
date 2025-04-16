def create_and_check_bert_for_sequence_classification(self, config,
    input_ids, token_type_ids, input_mask, sequence_labels, token_labels,
    choice_labels):
    config.num_labels = self.num_labels
    model = BertForSequenceClassification(config)
    model.eval()
    loss, logits = model(input_ids, attention_mask=input_mask,
        token_type_ids=token_type_ids, labels=sequence_labels)
    result = {'loss': loss, 'logits': logits}
    self.parent.assertListEqual(list(result['logits'].size()), [self.
        batch_size, self.num_labels])
    self.check_loss_output(result)
