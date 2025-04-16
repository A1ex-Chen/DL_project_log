def set_use_npu_flash_attention(self, use_npu_flash_attention: bool) ->None:
    """
        Set whether to use npu flash attention from `torch_npu` or not.

        """
    if use_npu_flash_attention:
        processor = AttnProcessorNPU()
    else:
        processor = AttnProcessor2_0() if hasattr(F,
            'scaled_dot_product_attention'
            ) and self.scale_qk else AttnProcessor()
    self.set_processor(processor)
