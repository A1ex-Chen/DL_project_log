def _eval_main_loop(self, label_tensor, hidden0, hidden1, time_idx,
    label_idx, complete_mask, num_symbol_added, num_total_symbol, current_label
    ):
    x, out_len, arange_tensor = self.stashed_tensor
    hidden = [hidden0, hidden1]
    time_idx_clapped = time_idx.data.clone()
    time_idx_clapped.masked_fill_(complete_mask, 0)
    f = x[arange_tensor, time_idx_clapped, :].unsqueeze(1)
    g, hidden_prime = self.model.predict_batch(current_label, hidden,
        add_sos=False)
    logp = self.model.joint(f, g)[:, 0, 0, :]
    v, k = logp.max(1)
    k = k.int()
    non_blank_mask = k != self.blank_idx
    current_label = current_label * ~non_blank_mask + k * non_blank_mask
    label_tensor[arange_tensor, label_idx] = label_tensor[arange_tensor,
        label_idx] * complete_mask + current_label * ~complete_mask
    for i in range(2):
        expand_mask = non_blank_mask.unsqueeze(0).unsqueeze(2).expand(hidden
            [0].size())
        hidden[i] = hidden[i] * ~expand_mask + hidden_prime[i] * expand_mask
    num_symbol_added += non_blank_mask
    num_total_symbol += non_blank_mask
    time_out_mask = num_total_symbol >= self.max_symbol_per_sample
    exceed_mask = num_symbol_added >= self.max_symbols
    advance_mask = (~non_blank_mask | exceed_mask) & ~complete_mask
    time_idx += advance_mask
    label_idx += non_blank_mask & ~time_out_mask
    num_symbol_added.masked_fill_(advance_mask, 0)
    complete_mask = (time_idx >= out_len) | time_out_mask
    batch_complete = complete_mask.all()
    return (label_tensor, hidden[0], hidden[1], time_idx, label_idx,
        complete_mask, batch_complete, num_symbol_added, num_total_symbol,
        current_label)
