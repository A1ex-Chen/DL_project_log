def apply_model(self, *args, **kwargs):
    if len(args) == 3:
        encoder_hidden_states = args[-1]
        args = args[:2]
    if kwargs.get('cond', None) is not None:
        encoder_hidden_states = kwargs.pop('cond')
    return self.model(*args, encoder_hidden_states=encoder_hidden_states,
        **kwargs).sample
