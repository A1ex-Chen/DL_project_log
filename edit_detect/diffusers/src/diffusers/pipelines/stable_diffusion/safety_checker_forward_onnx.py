@torch.no_grad()
def forward_onnx(self, clip_input: torch.Tensor, images: torch.Tensor):
    pooled_output = self.vision_model(clip_input)[1]
    image_embeds = self.visual_projection(pooled_output)
    special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
    cos_dist = cosine_distance(image_embeds, self.concept_embeds)
    adjustment = 0.0
    special_scores = (special_cos_dist - self.special_care_embeds_weights +
        adjustment)
    special_care = torch.any(special_scores > 0, dim=1)
    special_adjustment = special_care * 0.01
    special_adjustment = special_adjustment.unsqueeze(1).expand(-1,
        cos_dist.shape[1])
    concept_scores = (cos_dist - self.concept_embeds_weights +
        special_adjustment)
    has_nsfw_concepts = torch.any(concept_scores > 0, dim=1)
    images[has_nsfw_concepts] = 0.0
    return images, has_nsfw_concepts
