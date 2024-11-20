def cuda(self, dtype=torch.float16, use_xformers=False):
    self.to('cuda', dtype)
    if hasattr(self, 'image_proj_model'):
        self.image_proj_model.to(self.unet.device).to(self.unet.dtype)
    if use_xformers:
        if is_xformers_available():
            import xformers
            from packaging import version
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse('0.0.16'):
                logger.warning(
                    'xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.'
                    )
            self.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                'xformers is not available. Make sure it is installed correctly'
                )
