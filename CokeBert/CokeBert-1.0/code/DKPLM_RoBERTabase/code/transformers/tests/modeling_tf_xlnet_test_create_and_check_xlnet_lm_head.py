def create_and_check_xlnet_lm_head(self, config, input_ids_1, input_ids_2,
    input_ids_q, perm_mask, input_mask, target_mapping, segment_ids,
    lm_labels, sequence_labels, is_impossible_labels):
    model = TFXLNetLMHeadModel(config)
    inputs_1 = {'input_ids': input_ids_1, 'token_type_ids': segment_ids}
    all_logits_1, mems_1 = model(inputs_1)
    inputs_2 = {'input_ids': input_ids_2, 'mems': mems_1, 'token_type_ids':
        segment_ids}
    all_logits_2, mems_2 = model(inputs_2)
    inputs_3 = {'input_ids': input_ids_q, 'perm_mask': perm_mask,
        'target_mapping': target_mapping}
    logits, _ = model(inputs_3)
    result = {'mems_1': [mem.numpy() for mem in mems_1], 'all_logits_1':
        all_logits_1.numpy(), 'mems_2': [mem.numpy() for mem in mems_2],
        'all_logits_2': all_logits_2.numpy()}
    self.parent.assertListEqual(list(result['all_logits_1'].shape), [self.
        batch_size, self.seq_length, self.vocab_size])
    self.parent.assertListEqual(list(list(mem.shape) for mem in result[
        'mems_1']), [[self.seq_length, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
    self.parent.assertListEqual(list(result['all_logits_2'].shape), [self.
        batch_size, self.seq_length, self.vocab_size])
    self.parent.assertListEqual(list(list(mem.shape) for mem in result[
        'mems_2']), [[self.mem_len, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
