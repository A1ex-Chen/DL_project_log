def encode_img(self, img_list):
    image = img_list[0]
    img_list.pop(0)
    if isinstance(image, str):
        raw_image = Image.open(image).convert('RGB')
        image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
    elif isinstance(image, Image.Image):
        raw_image = image
        image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
    image_emb, _ = self.model.encode_img(image)
    img_list.append(image_emb)
