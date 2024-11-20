def create_and_check_xlnet_lm_head(self, config, input_ids_1, input_ids_2,
    input_ids_q, perm_mask, input_mask, target_mapping, segment_ids,
    lm_labels, sequence_labels, is_impossible_labels):
    model = XLNetLMHeadModel(config)
    model.eval()
    loss_1, all_logits_1, mems_1 = model(input_ids_1, token_type_ids=
        segment_ids, labels=lm_labels)
    loss_2, all_logits_2, mems_2 = model(input_ids_2, token_type_ids=
        segment_ids, labels=lm_labels, mems=mems_1)
    logits, _ = model(input_ids_q, perm_mask=perm_mask, target_mapping=
        target_mapping)
    result = {'loss_1': loss_1, 'mems_1': mems_1, 'all_logits_1':
        all_logits_1, 'loss_2': loss_2, 'mems_2': mems_2, 'all_logits_2':
        all_logits_2}
    self.parent.assertListEqual(list(result['loss_1'].size()), [])
    self.parent.assertListEqual(list(result['all_logits_1'].size()), [self.
        batch_size, self.seq_length, self.vocab_size])
    self.parent.assertListEqual(list(list(mem.size()) for mem in result[
        'mems_1']), [[self.seq_length, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
    self.parent.assertListEqual(list(result['loss_2'].size()), [])
    self.parent.assertListEqual(list(result['all_logits_2'].size()), [self.
        batch_size, self.seq_length, self.vocab_size])
    self.parent.assertListEqual(list(list(mem.size()) for mem in result[
        'mems_2']), [[self.mem_len, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
