def create_and_check_distilbert_for_question_answering(self, config,
    input_ids, input_mask, sequence_labels, token_labels, choice_labels):
    model = TFDistilBertForQuestionAnswering(config=config)
    inputs = {'input_ids': input_ids, 'attention_mask': input_mask}
    start_logits, end_logits = model(inputs)
    result = {'start_logits': start_logits.numpy(), 'end_logits':
        end_logits.numpy()}
    self.parent.assertListEqual(list(result['start_logits'].shape), [self.
        batch_size, self.seq_length])
    self.parent.assertListEqual(list(result['end_logits'].shape), [self.
        batch_size, self.seq_length])
