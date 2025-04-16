def create_and_check_bert_for_question_answering(self, config, input_ids,
    token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
    model = TFBertForQuestionAnswering(config=config)
    inputs = {'input_ids': input_ids, 'attention_mask': input_mask,
        'token_type_ids': token_type_ids}
    start_logits, end_logits = model(inputs)
    result = {'start_logits': start_logits.numpy(), 'end_logits':
        end_logits.numpy()}
    self.parent.assertListEqual(list(result['start_logits'].shape), [self.
        batch_size, self.seq_length])
    self.parent.assertListEqual(list(result['end_logits'].shape), [self.
        batch_size, self.seq_length])
