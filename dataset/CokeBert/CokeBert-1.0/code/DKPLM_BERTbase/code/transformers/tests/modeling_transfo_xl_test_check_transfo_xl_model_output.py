def check_transfo_xl_model_output(self, result):
    self.parent.assertListEqual(list(result['hidden_states_1'].size()), [
        self.batch_size, self.seq_length, self.hidden_size])
    self.parent.assertListEqual(list(result['hidden_states_2'].size()), [
        self.batch_size, self.seq_length, self.hidden_size])
    self.parent.assertListEqual(list(list(mem.size()) for mem in result[
        'mems_1']), [[self.mem_len, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
    self.parent.assertListEqual(list(list(mem.size()) for mem in result[
        'mems_2']), [[self.mem_len, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
