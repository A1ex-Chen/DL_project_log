def _master_params_to_model_params(self):
    stash = self._amp_stash
    if multi_tensor_applier.available:
        if len(stash.all_fp16_params) > 0:
            multi_tensor_applier(stash.multi_tensor_scale, stash.
                dummy_overflow_buf, [stash.all_fp32_from_fp16_params, stash
                .all_fp16_params], 1.0)
    else:
        for fp16_group, fp32_from_fp16_group in zip(stash.fp16_groups,
            stash.fp32_from_fp16_groups):
            master_params_to_model_params(fp16_group, fp32_from_fp16_group)
