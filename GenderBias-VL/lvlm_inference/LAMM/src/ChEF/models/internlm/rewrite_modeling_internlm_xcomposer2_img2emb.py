def img2emb(self, image_list):
    image_tensor, atts_img_list, img_target_list = [], [], []
    for image in image_list:
        img_embeds = self.vision_proj(self.vit(image.unsqueeze(0)))
        image_tensor.append(img_embeds)
        atts_img = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(
            img_embeds.device)
        atts_img_list.append(atts_img)
        img_target = torch.ones(img_embeds.size()[:2], dtype=torch.long).to(
            img_embeds.device) * -100
        img_target_list.append(img_target)
    image_tensor = torch.cat(image_tensor, dim=1)
    atts_img_list = torch.cat(atts_img_list, dim=1)
    img_target_list = torch.cat(img_target_list, dim=1)
    return image_tensor, atts_img_list, img_target_list
