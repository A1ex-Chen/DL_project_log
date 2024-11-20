def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_output,
    inst_idx_to_position_map, n_bm):
    """ Decode and update beam status, and then return active beam idx """

    def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if
            not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        return dec_partial_seq

    def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
        dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long,
            device=self.device)
        dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst *
            n_bm, 1)
        return dec_partial_pos

    def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm
        ):
        dec_output, *_ = self.model.decoder(dec_seq, dec_pos, src_seq,
            enc_output)
        dec_output = dec_output[:, -1, :]
        word_prob = F.log_softmax(self.model.tgt_word_prj(dec_output), dim=1)
        word_prob = word_prob.view(n_active_inst, n_bm, -1)
        return word_prob

    def collect_active_inst_idx_list(inst_beams, word_prob,
        inst_idx_to_position_map):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_inst_complete = inst_beams[inst_idx].advance(word_prob[
                inst_position])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]
        return active_inst_idx_list
    n_active_inst = len(inst_idx_to_position_map)
    dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
    dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
    word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output,
        n_active_inst, n_bm)
    active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams,
        word_prob, inst_idx_to_position_map)
    return active_inst_idx_list
