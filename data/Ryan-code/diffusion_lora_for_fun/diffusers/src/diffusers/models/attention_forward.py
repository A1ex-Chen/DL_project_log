def forward(self, hidden_states: torch.Tensor, *args, **kwargs) ->torch.Tensor:
    if len(args) > 0 or kwargs.get('scale', None) is not None:
        deprecation_message = (
            'The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`.'
            )
        deprecate('scale', '1.0.0', deprecation_message)
    for module in self.net:
        hidden_states = module(hidden_states)
    return hidden_states
