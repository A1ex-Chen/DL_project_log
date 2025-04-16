def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):
    dec_output, *_ = self.model.decoder(dec_seq, dec_pos, src_seq, enc_output)
    dec_output = dec_output[:, -1, :]
    word_prob = F.log_softmax(self.model.tgt_word_prj(dec_output), dim=1)
    word_prob = word_prob.view(n_active_inst, n_bm, -1)
    return word_prob
