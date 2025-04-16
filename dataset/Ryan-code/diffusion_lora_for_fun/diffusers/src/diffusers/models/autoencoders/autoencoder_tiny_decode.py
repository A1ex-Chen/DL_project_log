@apply_forward_hook
def decode(self, x: torch.Tensor, generator: Optional[torch.Generator]=None,
    return_dict: bool=True) ->Union[DecoderOutput, Tuple[torch.Tensor]]:
    if self.use_slicing and x.shape[0] > 1:
        output = [(self._tiled_decode(x_slice) if self.use_tiling else self
            .decoder(x)) for x_slice in x.split(1)]
        output = torch.cat(output)
    else:
        output = self._tiled_decode(x) if self.use_tiling else self.decoder(x)
    if not return_dict:
        return output,
    return DecoderOutput(sample=output)
