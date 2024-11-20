def _init_rope(self):
    if self.config.rope_scaling is None:
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
            max_position_embeddings=self.max_position_embeddings)
    else:
        scaling_type = self.config.rope_scaling['type']
        scaling_factor = self.config.rope_scaling['factor']
        if scaling_type == 'linear':
            self.rotary_emb = LlamaLinearScalingRotaryEmbedding(self.
                head_dim, max_position_embeddings=self.
                max_position_embeddings, scaling_factor=scaling_factor)
        elif scaling_type == 'dynamic':
            self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(self.
                head_dim, max_position_embeddings=self.
                max_position_embeddings, scaling_factor=scaling_factor)
        else:
            raise ValueError(f'Unknown RoPE scaling type {scaling_type}')
