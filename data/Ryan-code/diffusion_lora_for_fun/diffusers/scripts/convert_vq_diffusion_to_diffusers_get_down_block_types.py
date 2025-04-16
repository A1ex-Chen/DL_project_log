def get_down_block_types(original_encoder_config):
    attn_resolutions = coerce_attn_resolutions(original_encoder_config[
        'attn_resolutions'])
    num_resolutions = len(original_encoder_config['ch_mult'])
    resolution = coerce_resolution(original_encoder_config['resolution'])
    curr_res = resolution
    down_block_types = []
    for _ in range(num_resolutions):
        if curr_res in attn_resolutions:
            down_block_type = 'AttnDownEncoderBlock2D'
        else:
            down_block_type = 'DownEncoderBlock2D'
        down_block_types.append(down_block_type)
        curr_res = [(r // 2) for r in curr_res]
    return down_block_types
