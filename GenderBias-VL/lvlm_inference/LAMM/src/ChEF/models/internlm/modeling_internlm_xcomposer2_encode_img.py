def encode_img(self, image):
    if image is None:
        return None
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
        image = self.vis_processor(image).unsqueeze(0).to(self.device)
    else:
        assert isinstance(image, torch.Tensor)
    img_embeds, atts_img, img_target = self.img2emb(image)
    return img_embeds
