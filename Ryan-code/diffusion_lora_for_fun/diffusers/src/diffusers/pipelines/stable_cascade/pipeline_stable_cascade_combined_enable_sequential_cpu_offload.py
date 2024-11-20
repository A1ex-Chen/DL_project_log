def enable_sequential_cpu_offload(self, gpu_id: Optional[int]=None, device:
    Union[torch.device, str]='cuda'):
    """
        Offloads all models (`unet`, `text_encoder`, `vae`, and `safety checker` state dicts) to CPU using 🤗
        Accelerate, significantly reducing memory usage. Models are moved to a `torch.device('meta')` and loaded on a
        GPU only when their specific submodule's `forward` method is called. Offloading happens on a submodule basis.
        Memory savings are higher than using `enable_model_cpu_offload`, but performance is lower.
        """
    self.prior_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)
    self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=
        device)
