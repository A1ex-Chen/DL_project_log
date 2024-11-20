def decode_depth(self, depth_latent: torch.Tensor) ->torch.Tensor:
    """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
    depth_latent = depth_latent / self.depth_latent_scale_factor
    z = self.vae.post_quant_conv(depth_latent)
    stacked = self.vae.decoder(z)
    depth_mean = stacked.mean(dim=1, keepdim=True)
    return depth_mean
