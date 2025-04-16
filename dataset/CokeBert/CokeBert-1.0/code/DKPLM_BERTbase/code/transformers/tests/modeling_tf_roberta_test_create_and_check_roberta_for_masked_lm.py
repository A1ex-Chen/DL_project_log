def create_and_check_roberta_for_masked_lm(self, config, input_ids,
    token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
    model = TFRobertaForMaskedLM(config=config)
    prediction_scores = model([input_ids, input_mask, token_type_ids])[0]
    result = {'prediction_scores': prediction_scores.numpy()}
    self.parent.assertListEqual(list(result['prediction_scores'].shape), [
        self.batch_size, self.seq_length, self.vocab_size])
