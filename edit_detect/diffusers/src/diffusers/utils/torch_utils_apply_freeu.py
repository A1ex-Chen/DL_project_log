def apply_freeu(resolution_idx: int, hidden_states: 'torch.Tensor',
    res_hidden_states: 'torch.Tensor', **freeu_kwargs) ->Tuple[
    'torch.Tensor', 'torch.Tensor']:
    """Applies the FreeU mechanism as introduced in https:
    //arxiv.org/abs/2309.11497. Adapted from the official code repository: https://github.com/ChenyangSi/FreeU.

    Args:
        resolution_idx (`int`): Integer denoting the UNet block where FreeU is being applied.
        hidden_states (`torch.Tensor`): Inputs to the underlying block.
        res_hidden_states (`torch.Tensor`): Features from the skip block corresponding to the underlying block.
        s1 (`float`): Scaling factor for stage 1 to attenuate the contributions of the skip features.
        s2 (`float`): Scaling factor for stage 2 to attenuate the contributions of the skip features.
        b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
        b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
    """
    if resolution_idx == 0:
        num_half_channels = hidden_states.shape[1] // 2
        hidden_states[:, :num_half_channels] = hidden_states[:, :
            num_half_channels] * freeu_kwargs['b1']
        res_hidden_states = fourier_filter(res_hidden_states, threshold=1,
            scale=freeu_kwargs['s1'])
    if resolution_idx == 1:
        num_half_channels = hidden_states.shape[1] // 2
        hidden_states[:, :num_half_channels] = hidden_states[:, :
            num_half_channels] * freeu_kwargs['b2']
        res_hidden_states = fourier_filter(res_hidden_states, threshold=1,
            scale=freeu_kwargs['s2'])
    return hidden_states, res_hidden_states
