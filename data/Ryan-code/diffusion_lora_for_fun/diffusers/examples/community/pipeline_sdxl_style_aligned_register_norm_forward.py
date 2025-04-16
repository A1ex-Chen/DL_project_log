def register_norm_forward(norm_layer: Union[nn.GroupNorm, nn.LayerNorm]
    ) ->Union[nn.GroupNorm, nn.LayerNorm]:
    if not hasattr(norm_layer, 'orig_forward'):
        setattr(norm_layer, 'orig_forward', norm_layer.forward)
    orig_forward = norm_layer.orig_forward

    def forward_(hidden_states: torch.Tensor) ->torch.Tensor:
        n = hidden_states.shape[-2]
        hidden_states = concat_first(hidden_states, dim=-2)
        hidden_states = orig_forward(hidden_states)
        return hidden_states[..., :n, :]
    norm_layer.forward = forward_
    return norm_layer
