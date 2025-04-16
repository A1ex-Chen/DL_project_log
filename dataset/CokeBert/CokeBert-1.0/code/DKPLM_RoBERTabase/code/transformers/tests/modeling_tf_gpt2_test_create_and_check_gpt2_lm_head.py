def create_and_check_gpt2_lm_head(self, config, input_ids, input_mask,
    head_mask, token_type_ids, *args):
    model = TFGPT2LMHeadModel(config=config)
    inputs = {'input_ids': input_ids, 'attention_mask': input_mask,
        'token_type_ids': token_type_ids}
    prediction_scores = model(inputs)[0]
    result = {'prediction_scores': prediction_scores.numpy()}
    self.parent.assertListEqual(list(result['prediction_scores'].shape), [
        self.batch_size, self.seq_length, self.vocab_size])
