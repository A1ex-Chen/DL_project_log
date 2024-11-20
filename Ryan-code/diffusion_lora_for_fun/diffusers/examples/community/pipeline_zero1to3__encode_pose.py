def _encode_pose(self, pose, device, num_images_per_prompt,
    do_classifier_free_guidance):
    dtype = next(self.cc_projection.parameters()).dtype
    if isinstance(pose, torch.Tensor):
        pose_embeddings = pose.unsqueeze(1).to(device=device, dtype=dtype)
    else:
        if isinstance(pose[0], list):
            pose = torch.Tensor(pose)
        else:
            pose = torch.Tensor([pose])
        x, y, z = pose[:, 0].unsqueeze(1), pose[:, 1].unsqueeze(1), pose[:, 2
            ].unsqueeze(1)
        pose_embeddings = torch.cat([torch.deg2rad(x), torch.sin(torch.
            deg2rad(y)), torch.cos(torch.deg2rad(y)), z], dim=-1).unsqueeze(1
            ).to(device=device, dtype=dtype)
    bs_embed, seq_len, _ = pose_embeddings.shape
    pose_embeddings = pose_embeddings.repeat(1, num_images_per_prompt, 1)
    pose_embeddings = pose_embeddings.view(bs_embed * num_images_per_prompt,
        seq_len, -1)
    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(pose_embeddings)
        pose_embeddings = torch.cat([negative_prompt_embeds, pose_embeddings])
    return pose_embeddings
