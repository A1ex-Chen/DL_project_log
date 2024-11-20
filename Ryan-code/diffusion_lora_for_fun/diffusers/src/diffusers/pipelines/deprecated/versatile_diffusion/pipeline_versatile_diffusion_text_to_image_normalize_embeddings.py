def normalize_embeddings(encoder_output):
    embeds = self.text_encoder.text_projection(encoder_output.last_hidden_state
        )
    embeds_pooled = encoder_output.text_embeds
    embeds = embeds / torch.norm(embeds_pooled.unsqueeze(1), dim=-1,
        keepdim=True)
    return embeds
