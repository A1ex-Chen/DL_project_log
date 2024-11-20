def create_and_check_distilbert_for_masked_lm(self, config, input_ids,
    input_mask, sequence_labels, token_labels, choice_labels):
    model = TFDistilBertForMaskedLM(config=config)
    inputs = {'input_ids': input_ids, 'attention_mask': input_mask}
    prediction_scores, = model(inputs)
    result = {'prediction_scores': prediction_scores.numpy()}
    self.parent.assertListEqual(list(result['prediction_scores'].shape), [
        self.batch_size, self.seq_length, self.vocab_size])
