def extract_multimodal_feature(self, inputs):
    """Extract multimodal features from the input in Generation (Test)

        :param Dict inputs: input dict; modality: path
        :return _type_: _description_
        """
    features = []
    if 'image_paths' in inputs and inputs['image_paths']:
        image_embeds, _ = self.encode_image(inputs['image_paths'])
        features.append(image_embeds)
    if 'images' in inputs and inputs['images']:
        image_embeds, _ = self.encode_image_object(inputs['images'])
        return image_embeds
    if 'pcl_paths' in inputs and inputs['pcl_paths']:
        pcl_embeds, _ = self.encode_pcl(inputs['pcl_paths'])
        features.append(pcl_embeds)
    feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
    return feature_embeds
