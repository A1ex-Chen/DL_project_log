def _split(self, x, height, width):
    """
        Splits a flattened embedding x of shape (B, C * H * W + clip_img_dim) into two tensors of shape (B, C, H, W)
        and (B, 1, clip_img_dim)
        """
    batch_size = x.shape[0]
    latent_height = height // self.vae_scale_factor
    latent_width = width // self.vae_scale_factor
    img_vae_dim = self.num_channels_latents * latent_height * latent_width
    img_vae, img_clip = x.split([img_vae_dim, self.
        image_encoder_projection_dim], dim=1)
    img_vae = torch.reshape(img_vae, (batch_size, self.num_channels_latents,
        latent_height, latent_width))
    img_clip = torch.reshape(img_clip, (batch_size, 1, self.
        image_encoder_projection_dim))
    return img_vae, img_clip
