@apply_forward_hook
def encode(self, x: torch.Tensor, return_dict: bool=True) ->VQEncoderOutput:
    h = self.encoder(x)
    h = self.quant_conv(h)
    if not return_dict:
        return h,
    return VQEncoderOutput(latents=h)
