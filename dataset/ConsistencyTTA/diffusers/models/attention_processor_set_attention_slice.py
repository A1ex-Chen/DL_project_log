def set_attention_slice(self, slice_size):
    if slice_size is not None and slice_size > self.sliceable_head_dim:
        raise ValueError(
            f'slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.'
            )
    if slice_size is not None and self.added_kv_proj_dim is not None:
        processor = SlicedAttnAddedKVProcessor(slice_size)
    elif slice_size is not None:
        processor = SlicedAttnProcessor(slice_size)
    elif self.added_kv_proj_dim is not None:
        processor = AttnAddedKVProcessor()
    else:
        processor = AttnProcessor2_0() if hasattr(F,
            'scaled_dot_product_attention'
            ) and self.scale_qk else AttnProcessor()
    self.set_processor(processor)
