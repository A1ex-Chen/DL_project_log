def create_and_check_bert_for_pretraining(self, config, input_ids,
    token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
    model = TFBertForPreTraining(config=config)
    inputs = {'input_ids': input_ids, 'attention_mask': input_mask,
        'token_type_ids': token_type_ids}
    prediction_scores, seq_relationship_score = model(inputs)
    result = {'prediction_scores': prediction_scores.numpy(),
        'seq_relationship_score': seq_relationship_score.numpy()}
    self.parent.assertListEqual(list(result['prediction_scores'].shape), [
        self.batch_size, self.seq_length, self.vocab_size])
    self.parent.assertListEqual(list(result['seq_relationship_score'].shape
        ), [self.batch_size, 2])
