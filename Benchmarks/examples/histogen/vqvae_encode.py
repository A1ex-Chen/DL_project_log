def encode(self, input):
    enc_b = self.enc_b(input)
    enc_t = self.enc_t(enc_b)
    quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
    quant_t, diff_t, id_t = self.quantize_t(quant_t)
    quant_t = quant_t.permute(0, 3, 1, 2)
    diff_t = diff_t.unsqueeze(0)
    dec_t = self.dec_t(quant_t)
    enc_b = torch.cat([dec_t, enc_b], 1)
    quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
    quant_b, diff_b, id_b = self.quantize_b(quant_b)
    quant_b = quant_b.permute(0, 3, 1, 2)
    diff_b = diff_b.unsqueeze(0)
    return quant_t, quant_b, diff_t + diff_b, id_t, id_b
