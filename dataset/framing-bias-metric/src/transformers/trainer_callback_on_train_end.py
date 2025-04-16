def on_train_end(self, args, state, control, **kwargs):
    if state.is_local_process_zero:
        self.training_bar.close()
        self.training_bar = None
