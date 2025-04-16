def update_buffer(self):
    batch_size = self.infer_state['batch_size']
    self.b_seq_len += 1
    self.b_start_loc += torch.arange(0, batch_size, dtype=torch.int32,
        device=self.b_start_loc.device)
    return
