def _decode(self, z: torch.Tensor, image: Optional[torch.Tensor]=None, mask:
    Optional[torch.Tensor]=None, return_dict: bool=True) ->Union[
    DecoderOutput, Tuple[torch.Tensor]]:
    z = self.post_quant_conv(z)
    dec = self.decoder(z, image, mask)
    if not return_dict:
        return dec,
    return DecoderOutput(sample=dec)
