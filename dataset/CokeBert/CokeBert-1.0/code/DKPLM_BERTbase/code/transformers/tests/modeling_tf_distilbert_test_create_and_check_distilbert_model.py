def create_and_check_distilbert_model(self, config, input_ids, input_mask,
    sequence_labels, token_labels, choice_labels):
    model = TFDistilBertModel(config=config)
    inputs = {'input_ids': input_ids, 'attention_mask': input_mask}
    outputs = model(inputs)
    sequence_output = outputs[0]
    inputs = [input_ids, input_mask]
    sequence_output, = model(inputs)
    result = {'sequence_output': sequence_output.numpy()}
    self.parent.assertListEqual(list(result['sequence_output'].shape), [
        self.batch_size, self.seq_length, self.hidden_size])
