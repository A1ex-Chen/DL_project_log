def forward(self, input):
    quant_t, quant_b, diff, _, _ = self.encode(input)
    dec = self.decode(quant_t, quant_b)
    return dec, diff
