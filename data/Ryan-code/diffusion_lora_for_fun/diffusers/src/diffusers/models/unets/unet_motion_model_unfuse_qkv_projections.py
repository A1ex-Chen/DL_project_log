def unfuse_qkv_projections(self):
    """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
    if self.original_attn_processors is not None:
        self.set_attn_processor(self.original_attn_processors)
