@property
def _execution_device(self):
    """
        Returns the device on which the pipeline's models will be executed. After calling
        [`~DiffusionPipeline.enable_sequential_cpu_offload`] the execution device can only be inferred from
        Accelerate's module hooks.
        """
    for name, model in self.components.items():
        if not isinstance(model, torch.nn.Module
            ) or name in self._exclude_from_cpu_offload:
            continue
        if not hasattr(model, '_hf_hook'):
            return self.device
        for module in model.modules():
            if hasattr(module, '_hf_hook') and hasattr(module._hf_hook,
                'execution_device'
                ) and module._hf_hook.execution_device is not None:
                return torch.device(module._hf_hook.execution_device)
    return self.device
