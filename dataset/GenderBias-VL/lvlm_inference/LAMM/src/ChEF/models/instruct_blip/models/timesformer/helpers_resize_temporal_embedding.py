def resize_temporal_embedding(state_dict, key, num_frames):
    logging.info(
        f'Resizing temporal position embedding from {state_dict[key].size(1)} to {num_frames}'
        )
    time_embed = state_dict[key].transpose(1, 2)
    new_time_embed = F.interpolate(time_embed, size=num_frames, mode='nearest')
    return new_time_embed.transpose(1, 2)
