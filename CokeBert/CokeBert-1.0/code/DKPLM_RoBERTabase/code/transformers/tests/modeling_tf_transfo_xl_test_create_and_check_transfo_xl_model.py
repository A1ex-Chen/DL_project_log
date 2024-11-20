def create_and_check_transfo_xl_model(self, config, input_ids_1,
    input_ids_2, lm_labels):
    model = TFTransfoXLModel(config)
    hidden_states_1, mems_1 = model(input_ids_1)
    inputs = {'input_ids': input_ids_2, 'mems': mems_1}
    hidden_states_2, mems_2 = model(inputs)
    result = {'hidden_states_1': hidden_states_1.numpy(), 'mems_1': [mem.
        numpy() for mem in mems_1], 'hidden_states_2': hidden_states_2.
        numpy(), 'mems_2': [mem.numpy() for mem in mems_2]}
    self.parent.assertListEqual(list(result['hidden_states_1'].shape), [
        self.batch_size, self.seq_length, self.hidden_size])
    self.parent.assertListEqual(list(result['hidden_states_2'].shape), [
        self.batch_size, self.seq_length, self.hidden_size])
    self.parent.assertListEqual(list(list(mem.shape) for mem in result[
        'mems_1']), [[self.mem_len, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
    self.parent.assertListEqual(list(list(mem.shape) for mem in result[
        'mems_2']), [[self.mem_len, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
