def ensure_exactly_one_mask_token(self, masked_index: np.ndarray):
    numel = np.prod(masked_index.shape)
    if numel > 1:
        raise PipelineException('fill-mask', self.model.base_model_prefix,
            f'More than one mask_token ({self.tokenizer.mask_token}) is not supported'
            )
    elif numel < 1:
        raise PipelineException('fill-mask', self.model.base_model_prefix,
            f'No mask_token ({self.tokenizer.mask_token}) found on the input')
