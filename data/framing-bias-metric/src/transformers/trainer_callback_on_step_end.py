def on_step_end(self, args, state, control, **kwargs):
    if state.is_local_process_zero:
        self.training_bar.update(state.global_step - self.current_step)
        self.current_step = state.global_step
