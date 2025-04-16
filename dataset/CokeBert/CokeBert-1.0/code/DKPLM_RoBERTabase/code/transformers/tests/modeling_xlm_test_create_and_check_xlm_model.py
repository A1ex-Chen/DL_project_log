def create_and_check_xlm_model(self, config, input_ids, token_type_ids,
    input_lengths, sequence_labels, token_labels, is_impossible_labels,
    input_mask):
    model = XLMModel(config=config)
    model.eval()
    outputs = model(input_ids, lengths=input_lengths, langs=token_type_ids)
    outputs = model(input_ids, langs=token_type_ids)
    outputs = model(input_ids)
    sequence_output = outputs[0]
    result = {'sequence_output': sequence_output}
    self.parent.assertListEqual(list(result['sequence_output'].size()), [
        self.batch_size, self.seq_length, self.hidden_size])
