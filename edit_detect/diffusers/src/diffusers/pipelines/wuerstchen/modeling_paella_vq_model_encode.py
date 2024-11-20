@apply_forward_hook
def encode(self, x: torch.Tensor, return_dict: bool=True) ->VQEncoderOutput:
    h = self.in_block(x)
    h = self.down_blocks(h)
    if not return_dict:
        return h,
    return VQEncoderOutput(latents=h)
