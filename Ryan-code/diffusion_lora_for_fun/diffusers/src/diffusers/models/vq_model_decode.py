@apply_forward_hook
def decode(self, h: torch.Tensor, force_not_quantize: bool=False,
    return_dict: bool=True, shape=None) ->Union[DecoderOutput, torch.Tensor]:
    if not force_not_quantize:
        quant, _, _ = self.quantize(h)
    elif self.config.lookup_from_codebook:
        quant = self.quantize.get_codebook_entry(h, shape)
    else:
        quant = h
    quant2 = self.post_quant_conv(quant)
    dec = self.decoder(quant2, quant if self.config.norm_type == 'spatial' else
        None)
    if not return_dict:
        return dec,
    return DecoderOutput(sample=dec)
