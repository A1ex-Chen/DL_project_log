def img2emb(self, image):
    img_embeds = self.vision_proj(self.vit(image.to(self.device)))
    atts_img = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(
        img_embeds.device)
    img_target = torch.ones(img_embeds.size()[:2], dtype=torch.long).to(
        img_embeds.device) * -100
    return img_embeds, atts_img, img_target
