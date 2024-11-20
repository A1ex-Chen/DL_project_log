def create_and_check_xlnet_base_model(self, config, input_ids_1,
    input_ids_2, input_ids_q, perm_mask, input_mask, target_mapping,
    segment_ids, lm_labels, sequence_labels, is_impossible_labels):
    model = XLNetModel(config)
    model.eval()
    _, _ = model(input_ids_1, input_mask=input_mask)
    _, _ = model(input_ids_1, attention_mask=input_mask)
    _, _ = model(input_ids_1, token_type_ids=segment_ids)
    outputs, mems_1 = model(input_ids_1)
    result = {'mems_1': mems_1, 'outputs': outputs}
    config.mem_len = 0
    model = XLNetModel(config)
    model.eval()
    no_mems_outputs = model(input_ids_1)
    self.parent.assertEqual(len(no_mems_outputs), 1)
    self.parent.assertListEqual(list(result['outputs'].size()), [self.
        batch_size, self.seq_length, self.hidden_size])
    self.parent.assertListEqual(list(list(mem.size()) for mem in result[
        'mems_1']), [[self.seq_length, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
