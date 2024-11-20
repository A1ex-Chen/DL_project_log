def remove_all_hooks(self):
    """
        Removes all hooks that were added when using `enable_sequential_cpu_offload` or `enable_model_cpu_offload`.
        """
    for _, model in self.components.items():
        if isinstance(model, torch.nn.Module) and hasattr(model, '_hf_hook'):
            accelerate.hooks.remove_hook_from_module(model, recurse=True)
    self._all_hooks = []
