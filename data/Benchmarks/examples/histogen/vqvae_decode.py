def decode(self, quant_t, quant_b):
    upsample_t = self.upsample_t(quant_t)
    quant = torch.cat([upsample_t, quant_b], 1)
    dec = self.dec(quant)
    return dec
