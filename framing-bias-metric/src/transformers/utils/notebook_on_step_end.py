def on_step_end(self, args, state, control, **kwargs):
    epoch = int(state.epoch) if int(state.epoch
        ) == state.epoch else f'{state.epoch:.2f}'
    self.training_tracker.update(state.global_step + 1, comment=
        f'Epoch {epoch}/{state.num_train_epochs}', force_update=self.
        _force_next_update)
    self._force_next_update = False
