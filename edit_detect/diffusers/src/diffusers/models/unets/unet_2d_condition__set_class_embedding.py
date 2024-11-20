def _set_class_embedding(self, class_embed_type: Optional[str], act_fn: str,
    num_class_embeds: Optional[int], projection_class_embeddings_input_dim:
    Optional[int], time_embed_dim: int, timestep_input_dim: int):
    if class_embed_type is None and num_class_embeds is not None:
        self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
    elif class_embed_type == 'timestep':
        self.class_embedding = TimestepEmbedding(timestep_input_dim,
            time_embed_dim, act_fn=act_fn)
    elif class_embed_type == 'identity':
        self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
    elif class_embed_type == 'projection':
        if projection_class_embeddings_input_dim is None:
            raise ValueError(
                "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
        self.class_embedding = TimestepEmbedding(
            projection_class_embeddings_input_dim, time_embed_dim)
    elif class_embed_type == 'simple_projection':
        if projection_class_embeddings_input_dim is None:
            raise ValueError(
                "`class_embed_type`: 'simple_projection' requires `projection_class_embeddings_input_dim` be set"
                )
        self.class_embedding = nn.Linear(projection_class_embeddings_input_dim,
            time_embed_dim)
    else:
        self.class_embedding = None
