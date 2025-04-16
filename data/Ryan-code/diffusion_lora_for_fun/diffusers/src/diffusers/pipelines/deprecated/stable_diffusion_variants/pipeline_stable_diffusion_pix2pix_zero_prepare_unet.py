def prepare_unet(unet: UNet2DConditionModel):
    """Modifies the UNet (`unet`) to perform Pix2Pix Zero optimizations."""
    pix2pix_zero_attn_procs = {}
    for name in unet.attn_processors.keys():
        module_name = name.replace('.processor', '')
        module = unet.get_submodule(module_name)
        if 'attn2' in name:
            pix2pix_zero_attn_procs[name] = Pix2PixZeroAttnProcessor(
                is_pix2pix_zero=True)
            module.requires_grad_(True)
        else:
            pix2pix_zero_attn_procs[name] = Pix2PixZeroAttnProcessor(
                is_pix2pix_zero=False)
            module.requires_grad_(False)
    unet.set_attn_processor(pix2pix_zero_attn_procs)
    return unet
