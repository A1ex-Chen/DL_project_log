def encode_img(self, image):
    device = image.device
    if len(image.shape) > 4:
        image = image.reshape(-1, *image.shape[-3:])
    with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
        image_embeds = image_embeds[:, 1:, :]
        bs, pn, hs = image_embeds.shape
        image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))
        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(
            image.device)
    return inputs_llama, atts_llama
