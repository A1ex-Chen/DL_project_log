def create_and_check_bert_for_question_answering(self, config, input_ids,
    token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
    model = BertForQuestionAnswering(config=config)
    model.eval()
    loss, start_logits, end_logits = model(input_ids, attention_mask=
        input_mask, token_type_ids=token_type_ids, start_positions=
        sequence_labels, end_positions=sequence_labels)
    result = {'loss': loss, 'start_logits': start_logits, 'end_logits':
        end_logits}
    self.parent.assertListEqual(list(result['start_logits'].size()), [self.
        batch_size, self.seq_length])
    self.parent.assertListEqual(list(result['end_logits'].size()), [self.
        batch_size, self.seq_length])
    self.check_loss_output(result)
