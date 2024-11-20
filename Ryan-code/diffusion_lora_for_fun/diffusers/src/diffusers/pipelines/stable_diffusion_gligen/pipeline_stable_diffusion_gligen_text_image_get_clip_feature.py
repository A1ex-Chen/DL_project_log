def get_clip_feature(self, input, normalize_constant, device, is_image=False):
    """
        Get image and phrases embedding by using CLIP pretrain model. The image embedding is transformed into the
        phrases embedding space through a projection.
        """
    if is_image:
        if input is None:
            return None
        inputs = self.processor(images=[input], return_tensors='pt').to(device)
        inputs['pixel_values'] = inputs['pixel_values'].to(self.
            image_encoder.dtype)
        outputs = self.image_encoder(**inputs)
        feature = outputs.image_embeds
        feature = self.image_project(feature).squeeze(0)
        feature = feature / feature.norm() * normalize_constant
        feature = feature.unsqueeze(0)
    else:
        if input is None:
            return None
        inputs = self.tokenizer(input, return_tensors='pt', padding=True).to(
            device)
        outputs = self.text_encoder(**inputs)
        feature = outputs.pooler_output
    return feature
