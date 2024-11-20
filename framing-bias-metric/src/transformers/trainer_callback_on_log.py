def on_log(self, args, state, control, logs=None, **kwargs):
    _ = logs.pop('total_flos', None)
    if state.is_local_process_zero:
        print(logs)
