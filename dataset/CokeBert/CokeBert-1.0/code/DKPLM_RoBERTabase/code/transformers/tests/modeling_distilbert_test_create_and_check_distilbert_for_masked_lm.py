def create_and_check_distilbert_for_masked_lm(self, config, input_ids,
    input_mask, sequence_labels, token_labels, choice_labels):
    model = DistilBertForMaskedLM(config=config)
    model.eval()
    loss, prediction_scores = model(input_ids, attention_mask=input_mask,
        masked_lm_labels=token_labels)
    result = {'loss': loss, 'prediction_scores': prediction_scores}
    self.parent.assertListEqual(list(result['prediction_scores'].size()), [
        self.batch_size, self.seq_length, self.vocab_size])
    self.check_loss_output(result)
