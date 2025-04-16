def _get_add_time_ids(self, do_classifier_free_guidance=True):
    add_time_ids = [self.fps, self.motion_bucket_id, self.noise_aug_strength]
    passed_add_embed_dim = self.addition_time_embed_dim * len(add_time_ids)
    expected_add_embed_dim = self.addition_time_embed_dim * 3
    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f'Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`.'
            )
    add_time_ids = torch.tensor([add_time_ids], device=torch_device)
    add_time_ids = add_time_ids.repeat(1, 1)
    if do_classifier_free_guidance:
        add_time_ids = torch.cat([add_time_ids, add_time_ids])
    return add_time_ids
