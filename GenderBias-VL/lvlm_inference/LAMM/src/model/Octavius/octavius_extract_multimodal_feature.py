def extract_multimodal_feature(self, inputs):
    if 'images' in inputs and inputs['images']:
        self.vision_type = 'image'
        images = self.transform_vision_data(inputs['images'], self.device)
        image_embeds = self.encode_image(images)
        return image_embeds
    if 'image_paths' in inputs and inputs['image_paths']:
        self.vision_type = 'image'
        image_paths = inputs['image_paths']
        images = self.load_and_transform_image_data_clip(image_paths, self.
            device).to(self.llama_model.dtype)
        image_embeds = self.encode_image(images)
        return image_embeds
    features = []
    self.vision_type = 'pcl'
    pcl_embeds = self.encode_pcl(inputs)
    features.append(pcl_embeds)
    feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
    return feature_embeds
