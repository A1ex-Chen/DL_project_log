def create_and_check_xlnet_base_model(self, config, input_ids_1,
    input_ids_2, input_ids_q, perm_mask, input_mask, target_mapping,
    segment_ids, lm_labels, sequence_labels, is_impossible_labels):
    model = TFXLNetModel(config)
    inputs = {'input_ids': input_ids_1, 'input_mask': input_mask,
        'token_type_ids': segment_ids}
    _, _ = model(inputs)
    inputs = [input_ids_1, input_mask]
    outputs, mems_1 = model(inputs)
    result = {'mems_1': [mem.numpy() for mem in mems_1], 'outputs': outputs
        .numpy()}
    config.mem_len = 0
    model = TFXLNetModel(config)
    no_mems_outputs = model(inputs)
    self.parent.assertEqual(len(no_mems_outputs), 1)
    self.parent.assertListEqual(list(result['outputs'].shape), [self.
        batch_size, self.seq_length, self.hidden_size])
    self.parent.assertListEqual(list(list(mem.shape) for mem in result[
        'mems_1']), [[self.seq_length, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
