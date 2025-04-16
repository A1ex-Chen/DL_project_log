def inference_speed(self, *args, **kwargs) ->float:
    return separate_process_wrapper_fn(self._inference_speed, self.args.
        do_multi_processing)(*args, **kwargs)
