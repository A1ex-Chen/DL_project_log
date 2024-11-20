def interpolate_beit3(model, new_model_name):
    target_size = new_model_name.split('_')[-1]
    state_dict = model.state_dict()
    pos_embed_key = 'beit3.encoder.embed_positions.A.weight'
    pos_embed_checkpoint = state_dict[pos_embed_key]
    embedding_size = pos_embed_checkpoint.shape[-1]
    torchscale_model = True
    num_patches = model.beit3.vision_embed.num_patches
    num_extra_tokens = model.beit3.vision_embed.num_position_embeddings(
        ) + 2 - num_patches
    orig_size = int(num_patches ** 0.5)
    new_size = int(target_size) // 16
    if orig_size != new_size:
        print('Position interpolate from %dx%d to %dx%d' % (orig_size,
            orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:num_extra_tokens].unsqueeze(0)
        pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
            embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(
            new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        if torchscale_model:
            new_pos_embed = new_pos_embed.squeeze(0)
        state_dict[pos_embed_key] = new_pos_embed
    return state_dict
