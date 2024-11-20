def enable_sequential_cpu_offload(self, gpu_id: Optional[int]=None, device:
    Union[torch.device, str]='cuda'):
    """
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
    self.prior_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)
    self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=
        device)
