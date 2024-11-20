def normalize_embeddings(encoder_output):
    embeds = self.image_encoder.vision_model.post_layernorm(encoder_output.
        last_hidden_state)
    embeds = self.image_encoder.visual_projection(embeds)
    embeds_pooled = embeds[:, 0:1]
    embeds = embeds / torch.norm(embeds_pooled, dim=-1, keepdim=True)
    return embeds
