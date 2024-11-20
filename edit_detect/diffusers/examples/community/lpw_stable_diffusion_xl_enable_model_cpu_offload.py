def enable_model_cpu_offload(self, gpu_id=0):
    """
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
    if is_accelerate_available() and is_accelerate_version('>=', '0.17.0.dev0'
        ):
        from accelerate import cpu_offload_with_hook
    else:
        raise ImportError(
            '`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.'
            )
    device = torch.device(f'cuda:{gpu_id}')
    if self.device.type != 'cpu':
        self.to('cpu', silence_dtype_warnings=True)
        torch.cuda.empty_cache()
    model_sequence = [self.text_encoder, self.text_encoder_2
        ] if self.text_encoder is not None else [self.text_encoder_2]
    model_sequence.extend([self.unet, self.vae])
    hook = None
    for cpu_offloaded_model in model_sequence:
        _, hook = cpu_offload_with_hook(cpu_offloaded_model, device,
            prev_module_hook=hook)
    self.final_offload_hook = hook
