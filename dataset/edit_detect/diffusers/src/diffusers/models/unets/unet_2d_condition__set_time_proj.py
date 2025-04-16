def _set_time_proj(self, time_embedding_type: str, block_out_channels: int,
    flip_sin_to_cos: bool, freq_shift: float, time_embedding_dim: int) ->Tuple[
    int, int]:
    if time_embedding_type == 'fourier':
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
        if time_embed_dim % 2 != 0:
            raise ValueError(
                f'`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.'
                )
        self.time_proj = GaussianFourierProjection(time_embed_dim // 2,
            set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos)
        timestep_input_dim = time_embed_dim
    elif time_embedding_type == 'positional':
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos,
            freq_shift)
        timestep_input_dim = block_out_channels[0]
    else:
        raise ValueError(
            f'{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`.'
            )
    return time_embed_dim, timestep_input_dim
