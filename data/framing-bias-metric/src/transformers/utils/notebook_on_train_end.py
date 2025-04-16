def on_train_end(self, args, state, control, **kwargs):
    self.training_tracker.update(state.global_step, comment=
        f'Epoch {int(state.epoch)}/{state.num_train_epochs}', force_update=True
        )
    self.training_tracker = None
