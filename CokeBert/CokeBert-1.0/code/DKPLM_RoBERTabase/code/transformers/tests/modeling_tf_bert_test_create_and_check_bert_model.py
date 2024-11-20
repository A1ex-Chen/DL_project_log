def create_and_check_bert_model(self, config, input_ids, token_type_ids,
    input_mask, sequence_labels, token_labels, choice_labels):
    model = TFBertModel(config=config)
    inputs = {'input_ids': input_ids, 'attention_mask': input_mask,
        'token_type_ids': token_type_ids}
    sequence_output, pooled_output = model(inputs)
    inputs = [input_ids, input_mask]
    sequence_output, pooled_output = model(inputs)
    sequence_output, pooled_output = model(input_ids)
    result = {'sequence_output': sequence_output.numpy(), 'pooled_output':
        pooled_output.numpy()}
    self.parent.assertListEqual(list(result['sequence_output'].shape), [
        self.batch_size, self.seq_length, self.hidden_size])
    self.parent.assertListEqual(list(result['pooled_output'].shape), [self.
        batch_size, self.hidden_size])
