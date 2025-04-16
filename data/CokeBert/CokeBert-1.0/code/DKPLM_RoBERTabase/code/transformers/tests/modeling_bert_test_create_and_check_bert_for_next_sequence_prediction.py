def create_and_check_bert_for_next_sequence_prediction(self, config,
    input_ids, token_type_ids, input_mask, sequence_labels, token_labels,
    choice_labels):
    model = BertForNextSentencePrediction(config=config)
    model.eval()
    loss, seq_relationship_score = model(input_ids, attention_mask=
        input_mask, token_type_ids=token_type_ids, next_sentence_label=
        sequence_labels)
    result = {'loss': loss, 'seq_relationship_score': seq_relationship_score}
    self.parent.assertListEqual(list(result['seq_relationship_score'].size(
        )), [self.batch_size, 2])
    self.check_loss_output(result)
