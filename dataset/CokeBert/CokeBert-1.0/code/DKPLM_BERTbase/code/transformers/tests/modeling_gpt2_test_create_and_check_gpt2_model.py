def create_and_check_gpt2_model(self, config, input_ids, input_mask,
    head_mask, token_type_ids, *args):
    model = GPT2Model(config=config)
    model.eval()
    model(input_ids, token_type_ids=token_type_ids, head_mask=head_mask)
    model(input_ids, token_type_ids=token_type_ids)
    sequence_output, presents = model(input_ids)
    result = {'sequence_output': sequence_output, 'presents': presents}
    self.parent.assertListEqual(list(result['sequence_output'].size()), [
        self.batch_size, self.seq_length, self.hidden_size])
    self.parent.assertEqual(len(result['presents']), config.n_layer)
