def _get_add_time_ids(self, fps: int, motion_bucket_id: int,
    noise_aug_strength: float, dtype: torch.dtype, batch_size: int,
    num_videos_per_prompt: int, do_classifier_free_guidance: bool):
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
    passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(
        add_time_ids)
    expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f'Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`.'
            )
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)
    if do_classifier_free_guidance:
        add_time_ids = torch.cat([add_time_ids, add_time_ids])
    return add_time_ids
