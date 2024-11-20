@apply_forward_hook
def decode(self, z: torch.Tensor, generator: Optional[torch.Generator]=None,
    image: Optional[torch.Tensor]=None, mask: Optional[torch.Tensor]=None,
    return_dict: bool=True) ->Union[DecoderOutput, Tuple[torch.Tensor]]:
    decoded = self._decode(z, image, mask).sample
    if not return_dict:
        return decoded,
    return DecoderOutput(sample=decoded)
