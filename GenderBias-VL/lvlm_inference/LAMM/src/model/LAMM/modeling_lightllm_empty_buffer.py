def empty_buffer(self):
    assert self.infer_state is not None
    batch_size = self.infer_state['batch_size']
    max_input_len = self.infer_state['max_len_in_batch']
    for i in range(batch_size):
        self.model.base_model.mem_manager.free(self.b_loc[i, max_input_len -
            self.b_seq_len[i]:max_input_len])
    return
