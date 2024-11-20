def enable_model_cpu_offload(self, gpu_id: Optional[int]=None, device:
    Union[torch.device, str]='cuda'):
    """
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to "cuda"):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                default to "cuda".
        """
    is_pipeline_device_mapped = self.hf_device_map is not None and len(self
        .hf_device_map) > 1
    if is_pipeline_device_mapped:
        raise ValueError(
            "It seems like you have activated a device mapping strategy on the pipeline so calling `enable_model_cpu_offload() isn't allowed. You can call `reset_device_map()` first and then call `enable_model_cpu_offload()`."
            )
    if self.model_cpu_offload_seq is None:
        raise ValueError(
            'Model CPU offload cannot be enabled because no `model_cpu_offload_seq` class attribute is set.'
            )
    if is_accelerate_available() and is_accelerate_version('>=', '0.17.0.dev0'
        ):
        from accelerate import cpu_offload_with_hook
    else:
        raise ImportError(
            '`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.'
            )
    self.remove_all_hooks()
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
    self.to('cpu', silence_dtype_warnings=True)
    device_mod = getattr(torch, device.type, None)
    if hasattr(device_mod, 'empty_cache') and device_mod.is_available():
        device_mod.empty_cache()
    all_model_components = {k: v for k, v in self.components.items() if
        isinstance(v, torch.nn.Module)}
    self._all_hooks = []
    hook = None
    for model_str in self.model_cpu_offload_seq.split('->'):
        model = all_model_components.pop(model_str, None)
        if not isinstance(model, torch.nn.Module):
            continue
        _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)
        self._all_hooks.append(hook)
    for name, model in all_model_components.items():
        if not isinstance(model, torch.nn.Module):
            continue
        if name in self._exclude_from_cpu_offload:
            model.to(device)
        else:
            _, hook = cpu_offload_with_hook(model, device)
            self._all_hooks.append(hook)
