def fuse_qkv_projections(self, unet: bool=True, vae: bool=True):
    """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        Args:
            unet (`bool`, defaults to `True`): To apply fusion on the UNet.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.
        """
    self.fusing_unet = False
    self.fusing_vae = False
    if unet:
        self.fusing_unet = True
        self.unet.fuse_qkv_projections()
        self.unet.set_attn_processor(FusedAttnProcessor2_0())
    if vae:
        if not isinstance(self.vae, AutoencoderKL):
            raise ValueError(
                '`fuse_qkv_projections()` is only supported for the VAE of type `AutoencoderKL`.'
                )
        self.fusing_vae = True
        self.vae.fuse_qkv_projections()
        self.vae.set_attn_processor(FusedAttnProcessor2_0())
