def disable_freeu(self) ->None:
    """Disables the FreeU mechanism."""
    freeu_keys = {'s1', 's2', 'b1', 'b2'}
    for i, upsample_block in enumerate(self.up_blocks):
        for k in freeu_keys:
            if hasattr(upsample_block, k) or getattr(upsample_block, k, None
                ) is not None:
                setattr(upsample_block, k, None)
