def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
    dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not
        b.done]
    dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
    dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
    return dec_partial_seq
