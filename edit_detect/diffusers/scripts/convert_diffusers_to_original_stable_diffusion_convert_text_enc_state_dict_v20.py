def convert_text_enc_state_dict_v20(text_enc_dict):
    new_state_dict = {}
    capture_qkv_weight = {}
    capture_qkv_bias = {}
    for k, v in text_enc_dict.items():
        if k.endswith('.self_attn.q_proj.weight') or k.endswith(
            '.self_attn.k_proj.weight') or k.endswith(
            '.self_attn.v_proj.weight'):
            k_pre = k[:-len('.q_proj.weight')]
            k_code = k[-len('q_proj.weight')]
            if k_pre not in capture_qkv_weight:
                capture_qkv_weight[k_pre] = [None, None, None]
            capture_qkv_weight[k_pre][code2idx[k_code]] = v
            continue
        if k.endswith('.self_attn.q_proj.bias') or k.endswith(
            '.self_attn.k_proj.bias') or k.endswith('.self_attn.v_proj.bias'):
            k_pre = k[:-len('.q_proj.bias')]
            k_code = k[-len('q_proj.bias')]
            if k_pre not in capture_qkv_bias:
                capture_qkv_bias[k_pre] = [None, None, None]
            capture_qkv_bias[k_pre][code2idx[k_code]] = v
            continue
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(
            m.group(0))], k)
        new_state_dict[relabelled_key] = v
    for k_pre, tensors in capture_qkv_weight.items():
        if None in tensors:
            raise Exception(
                'CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing'
                )
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(
            m.group(0))], k_pre)
        new_state_dict[relabelled_key + '.in_proj_weight'] = torch.cat(tensors)
    for k_pre, tensors in capture_qkv_bias.items():
        if None in tensors:
            raise Exception(
                'CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing'
                )
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(
            m.group(0))], k_pre)
        new_state_dict[relabelled_key + '.in_proj_bias'] = torch.cat(tensors)
    return new_state_dict
