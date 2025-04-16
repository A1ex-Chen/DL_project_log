@apply_forward_hook
def decode(self, z: torch.FloatTensor, return_dict: bool=True) ->Union[
    DecoderOutput, torch.FloatTensor]:
    if self.use_slicing and z.shape[0] > 1:
        decoded_slices = [self._decode(z_slice).sample for z_slice in z.
            split(1)]
        decoded = torch.cat(decoded_slices)
    else:
        decoded = self._decode(z).sample
    if not return_dict:
        return decoded,
    return DecoderOutput(sample=decoded)
