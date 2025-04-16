def apply(self, pipe, i, t, callback_kwargs):
    self.state.append(callback_kwargs['latents'])
    return callback_kwargs
