def create_and_check_bert_for_pretraining(self, config, input_ids,
    token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
    model = BertForPreTraining(config=config)
    model.eval()
    loss, prediction_scores, seq_relationship_score = model(input_ids,
        attention_mask=input_mask, token_type_ids=token_type_ids,
        masked_lm_labels=token_labels, next_sentence_label=sequence_labels)
    result = {'loss': loss, 'prediction_scores': prediction_scores,
        'seq_relationship_score': seq_relationship_score}
    self.parent.assertListEqual(list(result['prediction_scores'].size()), [
        self.batch_size, self.seq_length, self.vocab_size])
    self.parent.assertListEqual(list(result['seq_relationship_score'].size(
        )), [self.batch_size, 2])
    self.check_loss_output(result)
