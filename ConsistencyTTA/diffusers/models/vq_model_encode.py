def encode(self, x: torch.FloatTensor, return_dict: bool=True
    ) ->VQEncoderOutput:
    h = self.encoder(x)
    h = self.quant_conv(h)
    if not return_dict:
        return h,
    return VQEncoderOutput(latents=h)
