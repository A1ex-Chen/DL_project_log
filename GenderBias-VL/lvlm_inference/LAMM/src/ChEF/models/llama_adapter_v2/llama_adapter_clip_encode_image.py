def clip_encode_image(self, x):
    x = self.clip.visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.
        zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
        x], dim=1)
    x = x + self.clip.visual.positional_embedding.to(x.dtype)
    x = self.clip.visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = self.clip.visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = self.clip.visual.ln_post(x[:, :, :])
    if self.clip.visual.proj is not None:
        x = x @ self.clip.visual.proj
    return x
