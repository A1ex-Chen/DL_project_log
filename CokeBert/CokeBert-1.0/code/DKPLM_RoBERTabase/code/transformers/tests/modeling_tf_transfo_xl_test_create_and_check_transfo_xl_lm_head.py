def create_and_check_transfo_xl_lm_head(self, config, input_ids_1,
    input_ids_2, lm_labels):
    model = TFTransfoXLLMHeadModel(config)
    lm_logits_1, mems_1 = model(input_ids_1)
    inputs = {'input_ids': input_ids_1, 'labels': lm_labels}
    _, mems_1 = model(inputs)
    lm_logits_2, mems_2 = model([input_ids_2, mems_1])
    inputs = {'input_ids': input_ids_1, 'mems': mems_1, 'labels': lm_labels}
    _, mems_2 = model(inputs)
    result = {'mems_1': [mem.numpy() for mem in mems_1], 'lm_logits_1':
        lm_logits_1.numpy(), 'mems_2': [mem.numpy() for mem in mems_2],
        'lm_logits_2': lm_logits_2.numpy()}
    self.parent.assertListEqual(list(result['lm_logits_1'].shape), [self.
        batch_size, self.seq_length, self.vocab_size])
    self.parent.assertListEqual(list(list(mem.shape) for mem in result[
        'mems_1']), [[self.mem_len, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
    self.parent.assertListEqual(list(result['lm_logits_2'].shape), [self.
        batch_size, self.seq_length, self.vocab_size])
    self.parent.assertListEqual(list(list(mem.shape) for mem in result[
        'mems_2']), [[self.mem_len, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
