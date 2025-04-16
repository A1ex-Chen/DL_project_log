def _disable_shared_attention_processors(self):
    """
        Helper method to disable usage of the Shared Attention Processor. All processors
        are reset to the default Attention Processor for pytorch versions above 2.0.
        """
    attn_procs = {}
    for i, name in enumerate(self.unet.attn_processors.keys()):
        attn_procs[name] = AttnProcessor2_0()
    self.unet.set_attn_processor(attn_procs)
