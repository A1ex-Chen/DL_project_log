def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=
    None, dec_enc_attn_mask=None):
    dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input,
        dec_input, mask=slf_attn_mask)
    dec_output *= non_pad_mask
    dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output,
        enc_output, mask=dec_enc_attn_mask)
    dec_output *= non_pad_mask
    dec_output = self.pos_ffn(dec_output)
    dec_output *= non_pad_mask
    return dec_output, dec_slf_attn, dec_enc_attn
