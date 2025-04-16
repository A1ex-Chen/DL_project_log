def fuse_qkv_projections(self):
    """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
    self.original_attn_processors = None
    for _, attn_processor in self.attn_processors.items():
        if 'Added' in str(attn_processor.__class__.__name__):
            raise ValueError(
                '`fuse_qkv_projections()` is not supported for models having added KV projections.'
                )
    self.original_attn_processors = self.attn_processors
    for module in self.modules():
        if isinstance(module, Attention):
            module.fuse_projections(fuse=True)
