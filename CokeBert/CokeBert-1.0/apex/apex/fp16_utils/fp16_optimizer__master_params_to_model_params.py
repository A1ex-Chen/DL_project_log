def _master_params_to_model_params(self):
    if multi_tensor_applier.available:
        if len(self.all_fp16_params) > 0:
            multi_tensor_applier(self.multi_tensor_scale, self.
                _dummy_overflow_buf, [self.all_fp32_from_fp16_params, self.
                all_fp16_params], 1.0)
    else:
        for fp16_group, fp32_from_fp16_group in zip(self.fp16_groups, self.
            fp32_from_fp16_groups):
            master_params_to_model_params(fp16_group, fp32_from_fp16_group)
