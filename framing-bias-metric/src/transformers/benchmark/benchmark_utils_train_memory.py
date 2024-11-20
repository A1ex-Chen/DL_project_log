def train_memory(self, *args, **kwargs) ->[Memory, Optional[MemorySummary]]:
    return separate_process_wrapper_fn(self._train_memory, self.args.
        do_multi_processing)(*args, **kwargs)
