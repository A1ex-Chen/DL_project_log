def create_and_check_xlm_model(self, config, input_ids, token_type_ids,
    input_lengths, sequence_labels, token_labels, is_impossible_labels,
    input_mask):
    model = TFXLMModel(config=config)
    inputs = {'input_ids': input_ids, 'lengths': input_lengths, 'langs':
        token_type_ids}
    outputs = model(inputs)
    inputs = [input_ids, input_mask]
    outputs = model(inputs)
    sequence_output = outputs[0]
    result = {'sequence_output': sequence_output.numpy()}
    self.parent.assertListEqual(list(result['sequence_output'].shape), [
        self.batch_size, self.seq_length, self.hidden_size])
