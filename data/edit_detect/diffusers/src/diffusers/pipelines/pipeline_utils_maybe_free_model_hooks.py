def maybe_free_model_hooks(self):
    """
        Function that offloads all components, removes all model hooks that were added when using
        `enable_model_cpu_offload` and then applies them again. In case the model has not been offloaded this function
        is a no-op. Make sure to add this function to the end of the `__call__` function of your pipeline so that it
        functions correctly when applying enable_model_cpu_offload.
        """
    if not hasattr(self, '_all_hooks') or len(self._all_hooks) == 0:
        return
    self.enable_model_cpu_offload(device=getattr(self, '_offload_device',
        'cuda'))
