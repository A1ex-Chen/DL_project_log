def create_and_check_xlm_qa(self, config, input_ids, token_type_ids,
    input_lengths, sequence_labels, token_labels, is_impossible_labels,
    input_mask):
    model = TFXLMForQuestionAnsweringSimple(config)
    inputs = {'input_ids': input_ids, 'lengths': input_lengths}
    outputs = model(inputs)
    start_logits, end_logits = model(inputs)
    result = {'start_logits': start_logits.numpy(), 'end_logits':
        end_logits.numpy()}
    self.parent.assertListEqual(list(result['start_logits'].shape), [self.
        batch_size, self.seq_length])
    self.parent.assertListEqual(list(result['end_logits'].shape), [self.
        batch_size, self.seq_length])
