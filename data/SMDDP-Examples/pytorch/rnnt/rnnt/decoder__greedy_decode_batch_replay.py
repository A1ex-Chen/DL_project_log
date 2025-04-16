def _greedy_decode_batch_replay(self, x, out_len):
    device = x.device
    B = x.size()[0]
    assert B <= self.cg_batch_size, 'this should not have happened'
    self.stashed_tensor[0][:x.size(0), :x.size(1)] = x
    self.stashed_tensor[1][:out_len.size(0)] = out_len
    hidden = [torch.zeros((2, self.cg_batch_size, self.rnnt_config[
        'pred_n_hid']), dtype=x.dtype, device=device)] * 2
    label_tensor = torch.zeros(self.cg_batch_size, self.
        max_symbol_per_sample, dtype=torch.int, device=device)
    current_label = torch.ones(self.cg_batch_size, dtype=torch.int, device=
        device) * -1
    time_idx = torch.zeros(self.cg_batch_size, dtype=torch.int64, device=device
        )
    label_idx = torch.zeros(self.cg_batch_size, dtype=torch.int64, device=
        device)
    complete_mask = time_idx >= self.stashed_tensor[1]
    batch_complete = complete_mask.all()
    num_symbol_added = torch.zeros(self.cg_batch_size, dtype=torch.int,
        device=device)
    num_total_symbol = torch.zeros(self.cg_batch_size, dtype=torch.int,
        device=device)
    arange_tensor = torch.arange(self.cg_batch_size, device=device)
    list_input_tensor = [label_tensor, hidden[0], hidden[1], time_idx,
        label_idx, complete_mask, num_symbol_added, num_total_symbol,
        current_label]
    while batch_complete == False:
        list_input_tensor = [label_tensor, hidden[0], hidden[1], time_idx,
            label_idx, complete_mask, num_symbol_added, num_total_symbol,
            current_label]
        (label_tensor, hidden[0], hidden[1], time_idx, label_idx,
            complete_mask, batch_complete, num_symbol_added,
            num_total_symbol, current_label) = self.main_loop_cg(*
            list_input_tensor)
    label = []
    for i in range(B):
        label.append(label_tensor[i, :label_idx[i]].tolist())
    return label
