@apply_forward_hook
def encode(self, x: torch.Tensor, return_dict: bool=True) ->Union[
    AutoencoderTinyOutput, Tuple[torch.Tensor]]:
    if self.use_slicing and x.shape[0] > 1:
        output = [(self._tiled_encode(x_slice) if self.use_tiling else self
            .encoder(x_slice)) for x_slice in x.split(1)]
        output = torch.cat(output)
    else:
        output = self._tiled_encode(x) if self.use_tiling else self.encoder(x)
    if not return_dict:
        return output,
    return AutoencoderTinyOutput(latents=output)
