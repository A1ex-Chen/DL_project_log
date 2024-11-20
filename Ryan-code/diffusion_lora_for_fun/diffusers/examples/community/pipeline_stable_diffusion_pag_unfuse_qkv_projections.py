def unfuse_qkv_projections(self, unet: bool=True, vae: bool=True):
    """Disable QKV projection fusion if enabled.
        <Tip warning={true}>
        This API is 🧪 experimental.
        </Tip>
        Args:
            unet (`bool`, defaults to `True`): To apply fusion on the UNet.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.
        """
    if unet:
        if not self.fusing_unet:
            logger.warning(
                'The UNet was not initially fused for QKV projections. Doing nothing.'
                )
        else:
            self.unet.unfuse_qkv_projections()
            self.fusing_unet = False
    if vae:
        if not self.fusing_vae:
            logger.warning(
                'The VAE was not initially fused for QKV projections. Doing nothing.'
                )
        else:
            self.vae.unfuse_qkv_projections()
            self.fusing_vae = False
