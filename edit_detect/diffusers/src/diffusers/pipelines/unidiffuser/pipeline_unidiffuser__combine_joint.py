def _combine_joint(self, img_vae, img_clip, text):
    """
        Combines a latent image img_vae of shape (B, C, H, W), a CLIP-embedded image img_clip of shape (B, L_img,
        clip_img_dim), and a text embedding text of shape (B, L_text, text_dim) into a single embedding x of shape (B,
        C * H * W + L_img * clip_img_dim + L_text * text_dim).
        """
    img_vae = torch.reshape(img_vae, (img_vae.shape[0], -1))
    img_clip = torch.reshape(img_clip, (img_clip.shape[0], -1))
    text = torch.reshape(text, (text.shape[0], -1))
    return torch.concat([img_vae, img_clip, text], dim=-1)
