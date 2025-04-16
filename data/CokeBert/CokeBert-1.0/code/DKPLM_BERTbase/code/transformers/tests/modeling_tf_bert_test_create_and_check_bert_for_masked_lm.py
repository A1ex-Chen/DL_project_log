def create_and_check_bert_for_masked_lm(self, config, input_ids,
    token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
    model = TFBertForMaskedLM(config=config)
    inputs = {'input_ids': input_ids, 'attention_mask': input_mask,
        'token_type_ids': token_type_ids}
    prediction_scores, = model(inputs)
    result = {'prediction_scores': prediction_scores.numpy()}
    self.parent.assertListEqual(list(result['prediction_scores'].shape), [
        self.batch_size, self.seq_length, self.vocab_size])
