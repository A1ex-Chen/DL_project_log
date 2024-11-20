def enable_sequential_cpu_offload(self, gpu_id=0):
    """
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae have their state dicts saved to CPU and then are moved to a `torch.device('meta') and loaded
        to GPU only when their specific submodule has its `forward` method called. Note that offloading happens on a
        submodule basis. Memory savings are higher than with `enable_model_cpu_offload`, but performance is lower.
        """
    if is_accelerate_available() and is_accelerate_version('>=', '0.14.0'):
        from accelerate import cpu_offload
    else:
        raise ImportError(
            '`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher'
            )
    device = torch.device(f'cuda:{gpu_id}')
    if self.device.type != 'cpu':
        self.to('cpu', silence_dtype_warnings=True)
        torch.cuda.empty_cache()
    for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
        cpu_offload(cpu_offloaded_model, device)
