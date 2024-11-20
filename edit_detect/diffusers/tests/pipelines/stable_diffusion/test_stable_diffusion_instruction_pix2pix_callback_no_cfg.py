def callback_no_cfg(pipe, i, t, callback_kwargs):
    if i == 1:
        for k, w in callback_kwargs.items():
            if k in self.callback_cfg_params:
                callback_kwargs[k] = callback_kwargs[k].chunk(3)[0]
        pipe._guidance_scale = 1.0
    return callback_kwargs
