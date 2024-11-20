def create_and_check_bert_model_for_masked_lm_as_decoder(self, config,
    input_ids, token_type_ids, input_mask, sequence_labels, token_labels,
    choice_labels, encoder_hidden_states, encoder_attention_mask):
    model = BertForMaskedLM(config=config)
    model.eval()
    loss, prediction_scores = model(input_ids, attention_mask=input_mask,
        token_type_ids=token_type_ids, masked_lm_labels=token_labels,
        encoder_hidden_states=encoder_hidden_states, encoder_attention_mask
        =encoder_attention_mask)
    loss, prediction_scores = model(input_ids, attention_mask=input_mask,
        token_type_ids=token_type_ids, masked_lm_labels=token_labels,
        encoder_hidden_states=encoder_hidden_states)
    result = {'loss': loss, 'prediction_scores': prediction_scores}
    self.parent.assertListEqual(list(result['prediction_scores'].size()), [
        self.batch_size, self.seq_length, self.vocab_size])
    self.check_loss_output(result)
