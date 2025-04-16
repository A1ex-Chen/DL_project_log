def callback_on_step_end(pipe, i, t, callback_kwargs):
    if i == interrupt_step_idx:
        pipe._interrupt = True
    return callback_kwargs
