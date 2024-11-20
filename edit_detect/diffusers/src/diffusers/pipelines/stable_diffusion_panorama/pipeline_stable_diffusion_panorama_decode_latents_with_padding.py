def decode_latents_with_padding(self, latents: torch.Tensor, padding: int=8
    ) ->torch.Tensor:
    """
        Decode the given latents with padding for circular inference.

        Args:
            latents (torch.Tensor): The input latents to decode.
            padding (int, optional): The number of latents to add on each side for padding. Defaults to 8.

        Returns:
            torch.Tensor: The decoded image with padding removed.

        Notes:
            - The padding is added to remove boundary artifacts and improve the output quality.
            - This would slightly increase the memory usage.
            - The padding pixels are then removed from the decoded image.

        """
    latents = 1 / self.vae.config.scaling_factor * latents
    latents_left = latents[..., :padding]
    latents_right = latents[..., -padding:]
    latents = torch.cat((latents_right, latents, latents_left), axis=-1)
    image = self.vae.decode(latents, return_dict=False)[0]
    padding_pix = self.vae_scale_factor * padding
    image = image[..., padding_pix:-padding_pix]
    return image
