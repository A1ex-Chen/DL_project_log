def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor,
    chunk_dim: int, chunk_size: int):
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f'`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`.'
            )
    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat([ff(hid_slice) for hid_slice in hidden_states.
        chunk(num_chunks, dim=chunk_dim)], dim=chunk_dim)
    return ff_output
