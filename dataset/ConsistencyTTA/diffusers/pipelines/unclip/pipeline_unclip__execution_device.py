@property
def _execution_device(self):
    """
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
    if self.device != torch.device('meta') or not hasattr(self.decoder,
        '_hf_hook'):
        return self.device
    for module in self.decoder.modules():
        if hasattr(module, '_hf_hook') and hasattr(module._hf_hook,
            'execution_device'
            ) and module._hf_hook.execution_device is not None:
            return torch.device(module._hf_hook.execution_device)
    return self.device
