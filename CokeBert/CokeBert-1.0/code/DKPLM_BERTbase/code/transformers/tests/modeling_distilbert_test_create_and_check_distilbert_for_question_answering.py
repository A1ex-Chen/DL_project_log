def create_and_check_distilbert_for_question_answering(self, config,
    input_ids, input_mask, sequence_labels, token_labels, choice_labels):
    model = DistilBertForQuestionAnswering(config=config)
    model.eval()
    loss, start_logits, end_logits = model(input_ids, attention_mask=
        input_mask, start_positions=sequence_labels, end_positions=
        sequence_labels)
    result = {'loss': loss, 'start_logits': start_logits, 'end_logits':
        end_logits}
    self.parent.assertListEqual(list(result['start_logits'].size()), [self.
        batch_size, self.seq_length])
    self.parent.assertListEqual(list(result['end_logits'].size()), [self.
        batch_size, self.seq_length])
    self.check_loss_output(result)
