def get_clip_image_emb(self, imgs):
    tensor_list = [self.clip_model_preprocess(img) for img in imgs]
    imgs = torch.stack(tensor_list)
    imgs = imgs.to(self.device)
    with torch.no_grad():
        image_features = self.clip_model.encode_image(imgs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features
