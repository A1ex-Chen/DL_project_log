def callback_increase_guidance(pipe, i, t, callback_kwargs):
    pipe._guidance_scale += 1.0
    return callback_kwargs
