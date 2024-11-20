def get_up_block_types(original_decoder_config):
    attn_resolutions = coerce_attn_resolutions(original_decoder_config[
        'attn_resolutions'])
    num_resolutions = len(original_decoder_config['ch_mult'])
    resolution = coerce_resolution(original_decoder_config['resolution'])
    curr_res = [(r // 2 ** (num_resolutions - 1)) for r in resolution]
    up_block_types = []
    for _ in reversed(range(num_resolutions)):
        if curr_res in attn_resolutions:
            up_block_type = 'AttnUpDecoderBlock2D'
        else:
            up_block_type = 'UpDecoderBlock2D'
        up_block_types.append(up_block_type)
        curr_res = [(r * 2) for r in curr_res]
    return up_block_types
