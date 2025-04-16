def set_default_attn_processor(self):
    """
        Disables custom attention processors and sets the default attention implementation.
        """
    if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.
        attn_processors.values()):
        processor = AttnProcessor()
    else:
        raise ValueError(
            f'Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}'
            )
    self.set_attn_processor(processor)
