def check_transfo_xl_lm_head_output(self, result):
    self.parent.assertListEqual(list(result['loss_1'].size()), [self.
        batch_size, self.seq_length])
    self.parent.assertListEqual(list(result['lm_logits_1'].size()), [self.
        batch_size, self.seq_length, self.vocab_size])
    self.parent.assertListEqual(list(list(mem.size()) for mem in result[
        'mems_1']), [[self.mem_len, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
    self.parent.assertListEqual(list(result['loss_2'].size()), [self.
        batch_size, self.seq_length])
    self.parent.assertListEqual(list(result['lm_logits_2'].size()), [self.
        batch_size, self.seq_length, self.vocab_size])
    self.parent.assertListEqual(list(list(mem.size()) for mem in result[
        'mems_2']), [[self.mem_len, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
