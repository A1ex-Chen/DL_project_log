def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
    dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long,
        device=self.device)
    dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst *
        n_bm, 1)
    return dec_partial_pos
