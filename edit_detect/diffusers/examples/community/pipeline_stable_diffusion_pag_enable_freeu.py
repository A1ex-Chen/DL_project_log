def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
    """Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.
        The suffixes after the scaling factors represent the stages where they are being applied.
        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.
        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
    if not hasattr(self, 'unet'):
        raise ValueError('The pipeline must have `unet` for using FreeU.')
    self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)
