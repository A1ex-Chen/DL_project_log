@classmethod
def _optionally_disable_offloading(cls, _pipeline):
    """
        Optionally removes offloading in case the pipeline has been already sequentially offloaded to CPU.

        Args:
            _pipeline (`DiffusionPipeline`):
                The pipeline to disable offloading for.

        Returns:
            tuple:
                A tuple indicating if `is_model_cpu_offload` or `is_sequential_cpu_offload` is True.
        """
    is_model_cpu_offload = False
    is_sequential_cpu_offload = False
    if _pipeline is not None and _pipeline.hf_device_map is None:
        for _, component in _pipeline.components.items():
            if isinstance(component, nn.Module) and hasattr(component,
                '_hf_hook'):
                if not is_model_cpu_offload:
                    is_model_cpu_offload = isinstance(component._hf_hook,
                        CpuOffload)
                if not is_sequential_cpu_offload:
                    is_sequential_cpu_offload = isinstance(component.
                        _hf_hook, AlignDevicesHook) or hasattr(component.
                        _hf_hook, 'hooks') and isinstance(component.
                        _hf_hook.hooks[0], AlignDevicesHook)
                logger.info(
                    'Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again.'
                    )
                remove_hook_from_module(component, recurse=
                    is_sequential_cpu_offload)
    return is_model_cpu_offload, is_sequential_cpu_offload
