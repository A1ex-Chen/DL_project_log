@staticmethod
def _reorder_cache(past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx:
    torch.Tensor) ->Tuple[Tuple[torch.Tensor]]:
    return tuple(tuple(past_state.index_select(0, beam_idx.to(past_state.
        device)) for past_state in layer_past) for layer_past in
        past_key_values)
