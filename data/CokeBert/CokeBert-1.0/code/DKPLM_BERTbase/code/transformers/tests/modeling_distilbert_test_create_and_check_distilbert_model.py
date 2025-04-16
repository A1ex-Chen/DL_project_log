def create_and_check_distilbert_model(self, config, input_ids, input_mask,
    sequence_labels, token_labels, choice_labels):
    model = DistilBertModel(config=config)
    model.eval()
    sequence_output, = model(input_ids, input_mask)
    sequence_output, = model(input_ids)
    result = {'sequence_output': sequence_output}
    self.parent.assertListEqual(list(result['sequence_output'].size()), [
        self.batch_size, self.seq_length, self.hidden_size])
