def forward(self, *, image_embeddings, prompt_embeds,
    text_encoder_hidden_states, do_classifier_free_guidance):
    if do_classifier_free_guidance:
        image_embeddings_batch_size = image_embeddings.shape[0]
        classifier_free_guidance_embeddings = (self.
            learned_classifier_free_guidance_embeddings.unsqueeze(0))
        classifier_free_guidance_embeddings = (
            classifier_free_guidance_embeddings.expand(
            image_embeddings_batch_size, -1))
        image_embeddings = torch.cat([classifier_free_guidance_embeddings,
            image_embeddings], dim=0)
    assert image_embeddings.shape[0] == prompt_embeds.shape[0]
    batch_size = prompt_embeds.shape[0]
    time_projected_prompt_embeds = self.embedding_proj(prompt_embeds)
    time_projected_image_embeddings = (self.
        clip_image_embeddings_project_to_time_embeddings(image_embeddings))
    additive_clip_time_embeddings = (time_projected_image_embeddings +
        time_projected_prompt_embeds)
    clip_extra_context_tokens = self.clip_extra_context_tokens_proj(
        image_embeddings)
    clip_extra_context_tokens = clip_extra_context_tokens.reshape(batch_size,
        -1, self.clip_extra_context_tokens)
    text_encoder_hidden_states = self.encoder_hidden_states_proj(
        text_encoder_hidden_states)
    text_encoder_hidden_states = self.text_encoder_hidden_states_norm(
        text_encoder_hidden_states)
    text_encoder_hidden_states = text_encoder_hidden_states.permute(0, 2, 1)
    text_encoder_hidden_states = torch.cat([clip_extra_context_tokens,
        text_encoder_hidden_states], dim=2)
    return text_encoder_hidden_states, additive_clip_time_embeddings
