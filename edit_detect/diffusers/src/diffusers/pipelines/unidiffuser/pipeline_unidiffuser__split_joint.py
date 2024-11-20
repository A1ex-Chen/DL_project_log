def _split_joint(self, x, height, width):
    """
        Splits a flattened embedding x of shape (B, C * H * W + clip_img_dim + text_seq_len * text_dim] into (img_vae,
        img_clip, text) where img_vae is of shape (B, C, H, W), img_clip is of shape (B, 1, clip_img_dim), and text is
        of shape (B, text_seq_len, text_dim).
        """
    batch_size = x.shape[0]
    latent_height = height // self.vae_scale_factor
    latent_width = width // self.vae_scale_factor
    img_vae_dim = self.num_channels_latents * latent_height * latent_width
    text_dim = self.text_encoder_seq_len * self.text_intermediate_dim
    img_vae, img_clip, text = x.split([img_vae_dim, self.
        image_encoder_projection_dim, text_dim], dim=1)
    img_vae = torch.reshape(img_vae, (batch_size, self.num_channels_latents,
        latent_height, latent_width))
    img_clip = torch.reshape(img_clip, (batch_size, 1, self.
        image_encoder_projection_dim))
    text = torch.reshape(text, (batch_size, self.text_encoder_seq_len, self
        .text_intermediate_dim))
    return img_vae, img_clip, text
