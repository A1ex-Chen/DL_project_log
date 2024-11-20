def enable_sequential_cpu_offload(self, gpu_id=0):
    """
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
    if is_accelerate_available():
        from accelerate import cpu_offload
    else:
        raise ImportError(
            'Please install accelerate via `pip install accelerate`')
    device = torch.device(f'cuda:{gpu_id}')
    for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
        if cpu_offloaded_model is not None:
            cpu_offload(cpu_offloaded_model, device)
