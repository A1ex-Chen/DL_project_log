def get_zero_embed(self, batch_size=1, device=None):
    device = device or self.device
    zero_img = torch.zeros(1, 3, self.image_encoder.config.image_size, self
        .image_encoder.config.image_size).to(device=device, dtype=self.
        image_encoder.dtype)
    zero_image_emb = self.image_encoder(zero_img)['image_embeds']
    zero_image_emb = zero_image_emb.repeat(batch_size, 1)
    return zero_image_emb
