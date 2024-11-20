def _init_rope(self):
    if self.config.rope_scaling is None:
        self.rotary_emb = InternLM2RotaryEmbedding(self.head_dim,
            max_position_embeddings=self.max_position_embeddings, base=self
            .config.rope_theta)
    else:
        scaling_type = self.config.rope_scaling['type']
        scaling_factor = self.config.rope_scaling['factor']
        if scaling_type == 'dynamic':
            self.rotary_emb = InternLM2DynamicNTKScalingRotaryEmbedding(self
                .head_dim, max_position_embeddings=self.
                max_position_embeddings, base=self.config.rope_theta,
                scaling_factor=scaling_factor)
        else:
            raise ValueError(
                "Currently we only support rotary embedding's type being 'dynamic'."
                )
    return self.rotary_emb
