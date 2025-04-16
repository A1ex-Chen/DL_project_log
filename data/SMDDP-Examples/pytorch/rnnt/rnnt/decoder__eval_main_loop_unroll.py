def _eval_main_loop_unroll(self, label_tensor, hidden0, hidden1, time_idx,
    label_idx, complete_mask, num_symbol_added, num_total_symbol, current_label
    ):
    for u in range(self.cg_unroll_factor):
        (label_tensor, hidden0, hidden1, time_idx, label_idx, complete_mask,
            batch_complete, num_symbol_added, num_total_symbol, current_label
            ) = (self._eval_main_loop_stream(label_tensor, hidden0, hidden1,
            time_idx, label_idx, complete_mask, num_symbol_added,
            num_total_symbol, current_label))
    return (label_tensor, hidden0, hidden1, time_idx, label_idx,
        complete_mask, batch_complete, num_symbol_added, num_total_symbol,
        current_label)
