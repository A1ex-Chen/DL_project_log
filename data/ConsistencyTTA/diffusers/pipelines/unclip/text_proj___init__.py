@register_to_config
def __init__(self, *, clip_extra_context_tokens: int=4, clip_embeddings_dim:
    int=768, time_embed_dim: int, cross_attention_dim):
    super().__init__()
    self.learned_classifier_free_guidance_embeddings = nn.Parameter(torch.
        zeros(clip_embeddings_dim))
    self.embedding_proj = nn.Linear(clip_embeddings_dim, time_embed_dim)
    self.clip_image_embeddings_project_to_time_embeddings = nn.Linear(
        clip_embeddings_dim, time_embed_dim)
    self.clip_extra_context_tokens = clip_extra_context_tokens
    self.clip_extra_context_tokens_proj = nn.Linear(clip_embeddings_dim, 
        self.clip_extra_context_tokens * cross_attention_dim)
    self.encoder_hidden_states_proj = nn.Linear(clip_embeddings_dim,
        cross_attention_dim)
    self.text_encoder_hidden_states_norm = nn.LayerNorm(cross_attention_dim)
