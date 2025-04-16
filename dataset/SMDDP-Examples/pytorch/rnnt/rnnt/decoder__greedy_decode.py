def _greedy_decode(self, model, x, out_len):
    training_state = model.training
    model.eval()
    device = x.device
    hidden = None
    label = []
    for time_idx in range(out_len):
        if self.max_symbol_per_sample is not None and len(label
            ) > self.max_symbol_per_sample:
            break
        f = x[time_idx, :, :].unsqueeze(0)
        not_blank = True
        symbols_added = 0
        while not_blank and (self.max_symbols is None or symbols_added <
            self.max_symbols):
            g, hidden_prime = self._pred_step(model, self._SOS if label ==
                [] else label[-1], hidden, device)
            logp = self._joint_step(model, f, g, log_normalize=False)[0, :]
            v, k = logp.max(0)
            k = k.item()
            if k == self.blank_idx:
                not_blank = False
            else:
                label.append(k)
                hidden = hidden_prime
            symbols_added += 1
    model.train(training_state)
    return label
