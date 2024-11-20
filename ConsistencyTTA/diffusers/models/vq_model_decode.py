def decode(self, h: torch.FloatTensor, force_not_quantize: bool=False,
    return_dict: bool=True) ->Union[DecoderOutput, torch.FloatTensor]:
    if not force_not_quantize:
        quant, emb_loss, info = self.quantize(h)
    else:
        quant = h
    quant = self.post_quant_conv(quant)
    dec = self.decoder(quant)
    if not return_dict:
        return dec,
    return DecoderOutput(sample=dec)
