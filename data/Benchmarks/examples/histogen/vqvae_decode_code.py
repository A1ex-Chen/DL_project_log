def decode_code(self, code_t, code_b):
    quant_t = self.quantize_t.embed_code(code_t)
    quant_t = quant_t.permute(0, 3, 1, 2)
    quant_b = self.quantize_b.embed_code(code_b)
    quant_b = quant_b.permute(0, 3, 1, 2)
    dec = self.decode(quant_t, quant_b)
    return dec
