def train_speed(self, *args, **kwargs) ->float:
    return separate_process_wrapper_fn(self._train_speed, self.args.
        do_multi_processing)(*args, **kwargs)
