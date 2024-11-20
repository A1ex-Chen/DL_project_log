def setup_lora(self):
    """ Set up Low-Rank Adaptation (LoRA)
        """
    logger.info('Setting up low-rank adaptation.')
    self.student_unet.requires_grad_(False)
    lora_attn_procs = {}
    for name in self.student_unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith('attn1.processor'
            ) else self.student_unet.config.cross_attention_dim
        if name.startswith('mid_block'):
            hidden_size = self.student_unet.config.block_out_channels[-1]
        elif name.startswith('up_blocks'):
            block_id = int(name[len('up_blocks.')])
            hidden_size = list(reversed(self.student_unet.config.
                block_out_channels))[block_id]
        elif name.startswith('down_blocks'):
            block_id = int(name[len('down_blocks.')])
            hidden_size = self.student_unet.config.block_out_channels[block_id]
        if self.lightweight:
            hidden_size = hidden_size * 255 // 256
        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim)
    self.student_unet.set_attn_processor(lora_attn_procs)
