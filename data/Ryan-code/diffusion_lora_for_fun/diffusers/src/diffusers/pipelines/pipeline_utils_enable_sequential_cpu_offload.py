def enable_sequential_cpu_offload(self, gpu_id: Optional[int]=None, device:
    Union[torch.device, str]='cuda'):
    """
        Offloads all models to CPU using 🤗 Accelerate, significantly reducing memory usage. When called, the state
        dicts of all `torch.nn.Module` components (except those in `self._exclude_from_cpu_offload`) are saved to CPU
        and then moved to `torch.device('meta')` and loaded to GPU only when their specific submodule has its `forward`
        method called. Offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to "cuda"):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                default to "cuda".
        """
    if is_accelerate_available() and is_accelerate_version('>=', '0.14.0'):
        from accelerate import cpu_offload
    else:
        raise ImportError(
            '`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher'
            )
    self.remove_all_hooks()
    is_pipeline_device_mapped = self.hf_device_map is not None and len(self
        .hf_device_map) > 1
    if is_pipeline_device_mapped:
        raise ValueError(
            "It seems like you have activated a device mapping strategy on the pipeline so calling `enable_sequential_cpu_offload() isn't allowed. You can call `reset_device_map()` first and then call `enable_sequential_cpu_offload()`."
            )
    torch_device = torch.device(device)
    device_index = torch_device.index
    if gpu_id is not None and device_index is not None:
        raise ValueError(
            f'You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of the device: `device`={torch_device.type}'
            )
    self._offload_gpu_id = gpu_id or torch_device.index or getattr(self,
        '_offload_gpu_id', 0)
    device_type = torch_device.type
    device = torch.device(f'{device_type}:{self._offload_gpu_id}')
    self._offload_device = device
    if self.device.type != 'cpu':
        self.to('cpu', silence_dtype_warnings=True)
        device_mod = getattr(torch, self.device.type, None)
        if hasattr(device_mod, 'empty_cache') and device_mod.is_available():
            device_mod.empty_cache()
    for name, model in self.components.items():
        if not isinstance(model, torch.nn.Module):
            continue
        if name in self._exclude_from_cpu_offload:
            model.to(device)
        else:
            offload_buffers = len(model._parameters) > 0
            cpu_offload(model, device, offload_buffers=offload_buffers)
