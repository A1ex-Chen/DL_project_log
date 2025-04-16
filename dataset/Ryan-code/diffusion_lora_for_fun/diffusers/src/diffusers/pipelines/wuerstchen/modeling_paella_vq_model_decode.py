@apply_forward_hook
def decode(self, h: torch.Tensor, force_not_quantize: bool=True,
    return_dict: bool=True) ->Union[DecoderOutput, torch.Tensor]:
    if not force_not_quantize:
        quant, _, _ = self.vquantizer(h)
    else:
        quant = h
    x = self.up_blocks(quant)
    dec = self.out_block(x)
    if not return_dict:
        return dec,
    return DecoderOutput(sample=dec)
