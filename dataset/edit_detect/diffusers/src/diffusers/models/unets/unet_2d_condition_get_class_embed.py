def get_class_embed(self, sample: torch.Tensor, class_labels: Optional[
    torch.Tensor]) ->Optional[torch.Tensor]:
    class_emb = None
    if self.class_embedding is not None:
        if class_labels is None:
            raise ValueError(
                'class_labels should be provided when num_class_embeds > 0')
        if self.config.class_embed_type == 'timestep':
            class_labels = self.time_proj(class_labels)
            class_labels = class_labels.to(dtype=sample.dtype)
        class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
    return class_emb
