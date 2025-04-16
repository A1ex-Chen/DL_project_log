def _check_inference_step(self, n_step: int):
    """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
    assert n_step >= 1
    if isinstance(self.scheduler, DDIMScheduler):
        if n_step < 10:
            logging.warning(
                f'Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference.'
                )
    elif isinstance(self.scheduler, LCMScheduler):
        if not 1 <= n_step <= 4:
            logging.warning(
                f'Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps.'
                )
    else:
        raise RuntimeError(
            f'Unsupported scheduler type: {type(self.scheduler)}')
