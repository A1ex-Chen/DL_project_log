def create_and_check_openai_gpt_model(self, config, input_ids, head_mask,
    token_type_ids, *args):
    model = OpenAIGPTModel(config=config)
    model.eval()
    model(input_ids, token_type_ids=token_type_ids, head_mask=head_mask)
    model(input_ids, token_type_ids=token_type_ids)
    sequence_output, = model(input_ids)
    result = {'sequence_output': sequence_output}
    self.parent.assertListEqual(list(result['sequence_output'].size()), [
        self.batch_size, self.seq_length, self.hidden_size])
