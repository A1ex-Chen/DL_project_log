def _attn(self, query, key, value, causal_mask=None, attention_mask=None,
    head_mask=None):
    device = query.device
    if self.use_cache_quantization:
        qk, qk_scale, qk_zero = key
        if self.use_cache_kernel and self.cache_kernels is not None:
            shape = query.shape[:-1] + (qk.shape[-2],)
            attn_weights = torch.zeros(shape, dtype=torch.float16, device=
                device)
            self.cache_kernels.vecquant8matmul_batched_faster_old(query.
                contiguous() if query.dtype == torch.float16 else query.to(
                torch.float16).contiguous(), qk.transpose(-1, -2).
                contiguous(), attn_weights, qk_scale.contiguous() if 
                qk_scale.dtype == torch.float16 else qk_scale.to(torch.
                float16).contiguous(), qk_zero.contiguous() if qk_zero.
                dtype == torch.float16 else qk_zero.to(torch.float16).
                contiguous())
        else:
            key = dequantize_cache_torch(qk, qk_scale, qk_zero)
            attn_weights = torch.matmul(query, key.transpose(-1, -2))
    else:
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
    if self.scale_attn_weights:
        if self.use_cache_quantization:
            size_temp = value[0].size(-1)
        else:
            size_temp = value.size(-1)
        attn_weights = attn_weights / size_temp ** 0.5
    mask_value = torch.finfo(attn_weights.dtype).min
    if causal_mask is not None:
        attn_weights = torch.where(causal_mask, attn_weights.to(
            attn_weights.dtype), mask_value)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    if self.softmax_in_fp32:
        attn_weights = nn.functional.softmax(attn_weights.float(), dim=-1)
    else:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.type(query.dtype)
    attn_weights = self.attn_dropout(attn_weights)
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
    if self.use_cache_quantization:
        qv, qv_scale, qv_zero = value
        if self.use_cache_kernel and self.cache_kernels is not None:
            shape = attn_weights.shape[:-1] + (query.shape[-1],)
            attn_output = torch.zeros(shape, dtype=torch.float16, device=device
                )
            self.cache_kernels.vecquant8matmul_batched_column_compression_faster_old(
                attn_weights.contiguous() if attn_weights.dtype == torch.
                float16 else attn_weights.to(torch.float16).contiguous(),
                qv.contiguous(), attn_output, qv_scale.contiguous() if 
                qv_scale.dtype == torch.float16 else qv_scale.to(torch.
                float16).contiguous(), qv_zero.contiguous() if qv_zero.
                dtype == torch.float16 else qv_zero.to(torch.float16).
                contiguous())
            if attn_output.dtype != query.dtype:
                attn_output = attn_output.to(query.dtype)
                attn_weights = attn_weights.to(query.dtype)
        else:
            value = dequantize_cache_torch(qv, qv_scale, qv_zero)
            attn_output = torch.matmul(attn_weights, value)
    else:
        attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)
    return attn_output, attn_weights
