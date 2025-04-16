def sag_masking(self, original_latents, attn_map, map_size, t, eps):
    bh, hw1, hw2 = attn_map.shape
    b, latent_channel, latent_h, latent_w = original_latents.shape
    h = self.unet.attention_head_dim
    if isinstance(h, list):
        h = h[-1]
    attn_map = attn_map.reshape(b, h, hw1, hw2)
    attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > 1.0
    attn_mask = attn_mask.reshape(b, map_size[0], map_size[1]).unsqueeze(1
        ).repeat(1, latent_channel, 1, 1).type(attn_map.dtype)
    attn_mask = F.interpolate(attn_mask, (latent_h, latent_w))
    degraded_latents = gaussian_blur_2d(original_latents, kernel_size=9,
        sigma=1.0)
    degraded_latents = degraded_latents * attn_mask + original_latents * (1 -
        attn_mask)
    degraded_latents = self.scheduler.add_noise(degraded_latents, noise=eps,
        timesteps=t)
    return degraded_latents
