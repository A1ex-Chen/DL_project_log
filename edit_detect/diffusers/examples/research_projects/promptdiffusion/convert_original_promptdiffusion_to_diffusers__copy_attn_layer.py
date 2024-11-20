def _copy_attn_layer(hf_attn_layer, pt_attn_layer):
    hf_attn_layer.q_proj.weight.data = pt_attn_layer.to_q.weight
    hf_attn_layer.k_proj.weight.data = pt_attn_layer.to_k.weight
    hf_attn_layer.v_proj.weight.data = pt_attn_layer.to_v.weight
    hf_attn_layer.out_proj.weight = pt_attn_layer.to_out.weight
    hf_attn_layer.out_proj.bias = pt_attn_layer.to_out.bias
