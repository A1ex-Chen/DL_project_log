def resize_spatial_embedding(state_dict, key, num_patches):
    logging.info(
        f'Resizing spatial position embedding from {state_dict[key].size(1)} to {num_patches + 1}'
        )
    pos_embed = state_dict[key]
    cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
    other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
    new_pos_embed = F.interpolate(other_pos_embed, size=num_patches, mode=
        'nearest')
    new_pos_embed = new_pos_embed.transpose(1, 2)
    new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
    return new_pos_embed
