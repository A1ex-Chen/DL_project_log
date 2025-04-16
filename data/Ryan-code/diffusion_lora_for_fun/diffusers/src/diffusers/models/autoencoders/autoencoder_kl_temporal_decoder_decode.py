@apply_forward_hook
def decode(self, z: torch.Tensor, num_frames: int, return_dict: bool=True
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
    batch_size = z.shape[0] // num_frames
    image_only_indicator = torch.zeros(batch_size, num_frames, dtype=z.
        dtype, device=z.device)
    decoded = self.decoder(z, num_frames=num_frames, image_only_indicator=
        image_only_indicator)
    if not return_dict:
        return decoded,
    return DecoderOutput(sample=decoded)
