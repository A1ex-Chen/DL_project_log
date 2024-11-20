@apply_forward_hook
def decode(self, z: torch.Tensor, return_dict: bool=True, generator=None
    ) ->Union[DecoderOutput, torch.Tensor]:
    """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
    if self.use_slicing and z.shape[0] > 1:
        decoded_slices = [self._decode(z_slice).sample for z_slice in z.
            split(1)]
        decoded = torch.cat(decoded_slices)
    else:
        decoded = self._decode(z).sample
    if not return_dict:
        return decoded,
    return DecoderOutput(sample=decoded)
