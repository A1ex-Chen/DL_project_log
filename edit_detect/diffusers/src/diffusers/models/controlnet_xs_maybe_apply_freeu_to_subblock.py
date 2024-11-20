def maybe_apply_freeu_to_subblock(hidden_states, res_h_base):
    if is_freeu_enabled:
        return apply_freeu(self.resolution_idx, hidden_states, res_h_base,
            s1=self.s1, s2=self.s2, b1=self.b1, b2=self.b2)
    else:
        return hidden_states, res_h_base
