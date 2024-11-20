def _greedy_decode_batch(self, x, out_len):
    device = x.device
    B = x.size()[0]
    hidden = None
    assert self.max_symbol_per_sample is not None, 'max_symbol_per_sample needs to be specified in order to use batch_eval'
    label_tensor = torch.zeros(B, self.max_symbol_per_sample, dtype=torch.
        int, device=device)
    current_label = torch.ones(B, dtype=torch.int, device=device) * -1
    time_idx = torch.zeros(B, dtype=torch.int64, device=device)
    label_idx = torch.zeros(B, dtype=torch.int64, device=device)
    complete_mask = time_idx >= out_len
    num_symbol_added = torch.zeros(B, dtype=torch.int, device=device)
    num_total_symbol = torch.zeros(B, dtype=torch.int, device=device)
    time_out_mask = torch.ones(B, dtype=torch.bool, device=device)
    arange_tensor = torch.arange(B, device=device)
    while complete_mask.sum().item() != B:
        time_idx_clapped = time_idx.data.clone()
        time_idx_clapped.masked_fill_(complete_mask, 0)
        f = x[arange_tensor, time_idx_clapped, :].unsqueeze(1)
        """ The above code is essentially doing """
        g, hidden_prime = self._pred_step_batch(current_label, hidden, device)
        """ To test the serial joint """
        logp = self._joint_step(self.model, f, g, log_normalize=False)
        v, k = logp.max(1)
        k = k.int()
        non_blank_mask = k != self.blank_idx
        current_label = current_label * ~non_blank_mask + k * non_blank_mask
        if hidden == None:
            hidden = [None, None]
            hidden[0] = torch.zeros_like(hidden_prime[0])
            hidden[1] = torch.zeros_like(hidden_prime[1])
        """ We might need to do the following dynamic resizing """
        label_tensor[arange_tensor, label_idx] = label_tensor[arange_tensor,
            label_idx] * complete_mask + current_label * ~complete_mask
        """ Following is for testing the normal way of generate label """
        for i in range(2):
            expand_mask = non_blank_mask.unsqueeze(0).unsqueeze(2).expand(
                hidden[0].size())
            hidden[i] = hidden[i] * ~expand_mask + hidden_prime[i
                ] * expand_mask
        num_symbol_added += non_blank_mask
        num_total_symbol += non_blank_mask
        if self.max_symbol_per_sample == None:
            time_out_mask = torch.zeros_like(complete_mask)
        else:
            time_out_mask = num_total_symbol >= self.max_symbol_per_sample
        exceed_mask = num_symbol_added >= self.max_symbols
        advance_mask = (~non_blank_mask | exceed_mask) & ~complete_mask
        time_idx += advance_mask
        label_idx += non_blank_mask & ~time_out_mask
        num_symbol_added.masked_fill_(advance_mask, 0)
        complete_mask = (time_idx >= out_len) | time_out_mask
    label = []
    for i in range(B):
        label.append(label_tensor[i, :label_idx[i]].tolist())
    return label
