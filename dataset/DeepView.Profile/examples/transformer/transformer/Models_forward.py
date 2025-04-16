def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
    tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]
    enc_output, *_ = self.encoder(src_seq, src_pos)
    dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
    if self.tgt_emb_prj_weight_sharing:
        seq_logit = torch.matmul(dec_output, self.decoder.tgt_word_emb.
            weight.t())
    else:
        seq_logit = self.tgt_word_prj(dec_output)
    seq_logit = seq_logit * self.x_logit_scale
    return seq_logit.view(-1, seq_logit.size(2))
