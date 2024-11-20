def _combine(self, img_vae, img_clip):
    """
        Combines a latent iamge img_vae of shape (B, C, H, W) and a CLIP-embedded image img_clip of shape (B, 1,
        clip_img_dim) into a single tensor of shape (B, C * H * W + clip_img_dim).
        """
    img_vae = torch.reshape(img_vae, (img_vae.shape[0], -1))
    img_clip = torch.reshape(img_clip, (img_clip.shape[0], -1))
    return torch.concat([img_vae, img_clip], dim=-1)
