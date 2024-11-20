def create_and_check_bert_for_next_sequence_prediction(self, config,
    input_ids, token_type_ids, input_mask, sequence_labels, token_labels,
    choice_labels):
    model = TFBertForNextSentencePrediction(config=config)
    inputs = {'input_ids': input_ids, 'attention_mask': input_mask,
        'token_type_ids': token_type_ids}
    seq_relationship_score, = model(inputs)
    result = {'seq_relationship_score': seq_relationship_score.numpy()}
    self.parent.assertListEqual(list(result['seq_relationship_score'].shape
        ), [self.batch_size, 2])
