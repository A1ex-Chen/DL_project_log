@torch.no_grad()
def forward(self, clip_input, images):
    pooled_output = self.vision_model(clip_input)[1]
    image_embeds = self.visual_projection(pooled_output)
    special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds
        ).cpu().float().numpy()
    cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().float(
        ).numpy()
    result = []
    batch_size = image_embeds.shape[0]
    for i in range(batch_size):
        result_img = {'special_scores': {}, 'special_care': [],
            'concept_scores': {}, 'bad_concepts': []}
        adjustment = 0.0
        for concept_idx in range(len(special_cos_dist[0])):
            concept_cos = special_cos_dist[i][concept_idx]
            concept_threshold = self.special_care_embeds_weights[concept_idx
                ].item()
            result_img['special_scores'][concept_idx] = round(concept_cos -
                concept_threshold + adjustment, 3)
            if result_img['special_scores'][concept_idx] > 0:
                result_img['special_care'].append({concept_idx, result_img[
                    'special_scores'][concept_idx]})
                adjustment = 0.01
        for concept_idx in range(len(cos_dist[0])):
            concept_cos = cos_dist[i][concept_idx]
            concept_threshold = self.concept_embeds_weights[concept_idx].item()
            result_img['concept_scores'][concept_idx] = round(concept_cos -
                concept_threshold + adjustment, 3)
            if result_img['concept_scores'][concept_idx] > 0:
                result_img['bad_concepts'].append(concept_idx)
        result.append(result_img)
    has_nsfw_concepts = [(len(res['bad_concepts']) > 0) for res in result]
    for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
        if has_nsfw_concept:
            if torch.is_tensor(images) or torch.is_tensor(images[0]):
                images[idx] = torch.zeros_like(images[idx])
            else:
                images[idx] = np.zeros(images[idx].shape)
    if any(has_nsfw_concepts):
        logger.warning(
            'Potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed.'
            )
    return images, has_nsfw_concepts
