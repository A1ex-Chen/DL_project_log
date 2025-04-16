def _get_add_time_ids(self, original_size, crops_coords_top_left,
    target_size, aesthetic_score, negative_aesthetic_score, dtype,
    text_encoder_projection_dim=None):
    if self.config.requires_aesthetics_score:
        add_time_ids = list(original_size + crops_coords_top_left + (
            aesthetic_score,))
        add_neg_time_ids = list(original_size + crops_coords_top_left + (
            negative_aesthetic_score,))
    else:
        add_time_ids = list(original_size + crops_coords_top_left + target_size
            )
        add_neg_time_ids = list(original_size + crops_coords_top_left +
            target_size)
    passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(
        add_time_ids) + text_encoder_projection_dim
    expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
    if (expected_add_embed_dim > passed_add_embed_dim and 
        expected_add_embed_dim - passed_add_embed_dim == self.unet.config.
        addition_time_embed_dim):
        raise ValueError(
            f'Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model.'
            )
    elif expected_add_embed_dim < passed_add_embed_dim and passed_add_embed_dim - expected_add_embed_dim == self.unet.config.addition_time_embed_dim:
        raise ValueError(
            f'Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model.'
            )
    elif expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f'Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`.'
            )
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)
    return add_time_ids, add_neg_time_ids
