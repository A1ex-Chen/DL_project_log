def _get_add_time_ids(self, original_size, crops_coords_top_left,
    target_size, dtype, text_encoder_projection_dim=None):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(
        add_time_ids) + text_encoder_projection_dim
    expected_add_embed_dim = self.unet.base_add_embedding.linear_1.in_features
    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f'Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`.'
            )
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids
