def create_and_check_roberta_model(self, config, input_ids, token_type_ids,
    input_mask, sequence_labels, token_labels, choice_labels):
    model = TFRobertaModel(config=config)
    inputs = {'input_ids': input_ids, 'attention_mask': input_mask,
        'token_type_ids': token_type_ids}
    sequence_output = model(inputs)[0]
    inputs = [input_ids, input_mask]
    sequence_output = model(inputs)[0]
    sequence_output = model(input_ids)[0]
    result = {'sequence_output': sequence_output.numpy()}
    self.parent.assertListEqual(list(result['sequence_output'].shape), [
        self.batch_size, self.seq_length, self.hidden_size])
