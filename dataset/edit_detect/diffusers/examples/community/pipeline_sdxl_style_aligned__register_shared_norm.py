def _register_shared_norm(self, share_group_norm: bool=True,
    share_layer_norm: bool=True):
    """Helper method to register shared group/layer normalization layers."""

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

    def get_norm_layers(pipeline_, norm_layers_: Dict[str, List[Union[nn.
        GroupNorm, nn.LayerNorm]]]):
        if isinstance(pipeline_, nn.LayerNorm) and share_layer_norm:
            norm_layers_['layer'].append(pipeline_)
        if isinstance(pipeline_, nn.GroupNorm) and share_group_norm:
            norm_layers_['group'].append(pipeline_)
        else:
            for layer in pipeline_.children():
                get_norm_layers(layer, norm_layers_)
    norm_layers = {'group': [], 'layer': []}
    get_norm_layers(self.unet, norm_layers)
    norm_layers_list = []
    for key in ['group', 'layer']:
        for layer in norm_layers[key]:
            norm_layers_list.append(register_norm_forward(layer))
    return norm_layers_list
