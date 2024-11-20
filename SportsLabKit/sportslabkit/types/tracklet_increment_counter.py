def increment_counter(self, global_step: (int | None)=None) ->None:
    """Increment the step counters, steps_alive and global_step. If global_step is provided, it will be used instead of incrementing the global_step counter."""
    self.steps_alive += 1
    if global_step is not None:
        self.global_step = int(global_step)
    else:
        self.global_step += 1
