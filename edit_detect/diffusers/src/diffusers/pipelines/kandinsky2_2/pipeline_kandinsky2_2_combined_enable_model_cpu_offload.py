def enable_model_cpu_offload(self, gpu_id: Optional[int]=None, device:
    Union[torch.device, str]='cuda'):
    """
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
    self.prior_pipe.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
    self.decoder_pipe.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
