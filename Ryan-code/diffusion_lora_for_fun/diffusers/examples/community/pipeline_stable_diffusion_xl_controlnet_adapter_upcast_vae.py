def upcast_vae(self):
    dtype = self.vae.dtype
    self.vae.to(dtype=torch.float32)
    use_torch_2_0_or_xformers = isinstance(self.vae.decoder.mid_block.
        attentions[0].processor, (AttnProcessor2_0, XFormersAttnProcessor,
        LoRAXFormersAttnProcessor, LoRAAttnProcessor2_0))
    if use_torch_2_0_or_xformers:
        self.vae.post_quant_conv.to(dtype)
        self.vae.decoder.conv_in.to(dtype)
        self.vae.decoder.mid_block.to(dtype)
