def create_and_check_distilbert_for_token_classification(self, config,
    input_ids, input_mask, sequence_labels, token_labels, choice_labels):
    config.num_labels = self.num_labels
    model = DistilBertForTokenClassification(config=config)
    model.eval()
    loss, logits = model(input_ids, attention_mask=input_mask, labels=
        token_labels)
    result = {'loss': loss, 'logits': logits}
    self.parent.assertListEqual(list(result['logits'].size()), [self.
        batch_size, self.seq_length, self.num_labels])
    self.check_loss_output(result)
