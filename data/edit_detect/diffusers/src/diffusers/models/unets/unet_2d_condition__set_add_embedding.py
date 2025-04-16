def _set_add_embedding(self, addition_embed_type: str,
    addition_embed_type_num_heads: int, addition_time_embed_dim: Optional[
    int], flip_sin_to_cos: bool, freq_shift: float, cross_attention_dim:
    Optional[int], encoder_hid_dim: Optional[int],
    projection_class_embeddings_input_dim: Optional[int], time_embed_dim: int):
    if addition_embed_type == 'text':
        if encoder_hid_dim is not None:
            text_time_embedding_from_dim = encoder_hid_dim
        else:
            text_time_embedding_from_dim = cross_attention_dim
        self.add_embedding = TextTimeEmbedding(text_time_embedding_from_dim,
            time_embed_dim, num_heads=addition_embed_type_num_heads)
    elif addition_embed_type == 'text_image':
        self.add_embedding = TextImageTimeEmbedding(text_embed_dim=
            cross_attention_dim, image_embed_dim=cross_attention_dim,
            time_embed_dim=time_embed_dim)
    elif addition_embed_type == 'text_time':
        self.add_time_proj = Timesteps(addition_time_embed_dim,
            flip_sin_to_cos, freq_shift)
        self.add_embedding = TimestepEmbedding(
            projection_class_embeddings_input_dim, time_embed_dim)
    elif addition_embed_type == 'image':
        self.add_embedding = ImageTimeEmbedding(image_embed_dim=
            encoder_hid_dim, time_embed_dim=time_embed_dim)
    elif addition_embed_type == 'image_hint':
        self.add_embedding = ImageHintTimeEmbedding(image_embed_dim=
            encoder_hid_dim, time_embed_dim=time_embed_dim)
    elif addition_embed_type is not None:
        raise ValueError(
            f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'."
            )
