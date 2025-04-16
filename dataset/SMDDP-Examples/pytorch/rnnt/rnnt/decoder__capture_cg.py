def _capture_cg(self, x, out_len):
    self.model.eval()
    device = x.device
    B = x.size()[0]
    self.cg_batch_size = B
    hidden = [torch.zeros((self.rnnt_config['pred_rnn_layers'], B, self.
        rnnt_config['pred_n_hid']), dtype=x.dtype, device=device)] * 2
    assert self.max_symbol_per_sample is not None, 'max_symbol_per_sample needs to be specified in order to use batch_eval'
    label_tensor = torch.zeros(B, self.max_symbol_per_sample, dtype=torch.
        int, device=device)
    current_label = torch.ones(B, dtype=torch.int, device=device) * -1
    time_idx = torch.zeros(B, dtype=torch.int64, device=device)
    label_idx = torch.zeros(B, dtype=torch.int64, device=device)
    complete_mask = time_idx >= out_len
    num_symbol_added = torch.zeros(B, dtype=torch.int, device=device)
    num_total_symbol = torch.zeros(B, dtype=torch.int, device=device)
    arange_tensor = torch.arange(B, device=device)
    list_input_tensor = [label_tensor, hidden[0], hidden[1], time_idx,
        label_idx, complete_mask, num_symbol_added, num_total_symbol,
        current_label]
    self.stashed_tensor = x, out_len, arange_tensor
    self.main_loop_cg = self._capture_cg_for_main_loop(list_input_tensor)
    self.cg_captured = True
