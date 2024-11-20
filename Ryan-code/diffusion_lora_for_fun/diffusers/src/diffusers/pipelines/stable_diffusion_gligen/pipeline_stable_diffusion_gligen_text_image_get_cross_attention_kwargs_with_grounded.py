def get_cross_attention_kwargs_with_grounded(self, hidden_size,
    gligen_phrases, gligen_images, gligen_boxes, input_phrases_mask,
    input_images_mask, repeat_batch, normalize_constant, max_objs, device):
    """
        Prepare the cross-attention kwargs containing information about the grounded input (boxes, mask, image
        embedding, phrases embedding).
        """
    phrases, images = gligen_phrases, gligen_images
    images = [None] * len(phrases) if images is None else images
    phrases = [None] * len(images) if phrases is None else phrases
    boxes = torch.zeros(max_objs, 4, device=device, dtype=self.text_encoder
        .dtype)
    masks = torch.zeros(max_objs, device=device, dtype=self.text_encoder.dtype)
    phrases_masks = torch.zeros(max_objs, device=device, dtype=self.
        text_encoder.dtype)
    image_masks = torch.zeros(max_objs, device=device, dtype=self.
        text_encoder.dtype)
    phrases_embeddings = torch.zeros(max_objs, hidden_size, device=device,
        dtype=self.text_encoder.dtype)
    image_embeddings = torch.zeros(max_objs, hidden_size, device=device,
        dtype=self.text_encoder.dtype)
    text_features = []
    image_features = []
    for phrase, image in zip(phrases, images):
        text_features.append(self.get_clip_feature(phrase,
            normalize_constant, device, is_image=False))
        image_features.append(self.get_clip_feature(image,
            normalize_constant, device, is_image=True))
    for idx, (box, text_feature, image_feature) in enumerate(zip(
        gligen_boxes, text_features, image_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            phrases_embeddings[idx] = text_feature
            phrases_masks[idx] = 1
        if image_feature is not None:
            image_embeddings[idx] = image_feature
            image_masks[idx] = 1
    input_phrases_mask = self.complete_mask(input_phrases_mask, max_objs,
        device)
    phrases_masks = phrases_masks.unsqueeze(0).repeat(repeat_batch, 1
        ) * input_phrases_mask
    input_images_mask = self.complete_mask(input_images_mask, max_objs, device)
    image_masks = image_masks.unsqueeze(0).repeat(repeat_batch, 1
        ) * input_images_mask
    boxes = boxes.unsqueeze(0).repeat(repeat_batch, 1, 1)
    masks = masks.unsqueeze(0).repeat(repeat_batch, 1)
    phrases_embeddings = phrases_embeddings.unsqueeze(0).repeat(repeat_batch,
        1, 1)
    image_embeddings = image_embeddings.unsqueeze(0).repeat(repeat_batch, 1, 1)
    out = {'boxes': boxes, 'masks': masks, 'phrases_masks': phrases_masks,
        'image_masks': image_masks, 'phrases_embeddings':
        phrases_embeddings, 'image_embeddings': image_embeddings}
    return out
