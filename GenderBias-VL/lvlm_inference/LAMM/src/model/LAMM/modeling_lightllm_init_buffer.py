def init_buffer(self, per_input_len, max_input_len, max_output_len):
    batch_size = self.infer_state['batch_size']
    self.b_loc = torch.zeros(batch_size, max_input_len + max_output_len,
        dtype=torch.long, device='cuda')
    self.b_seq_len = torch.as_tensor(per_input_len, dtype=torch.int32,
        device='cuda')
    self.b_start_loc = torch.cumsum(torch.as_tensor([0] + per_input_len[:-1
        ], dtype=torch.int32, device='cuda'), dim=0)
    return
