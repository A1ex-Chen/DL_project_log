def _copy_layer(hf_layer, pt_layer):
    _copy_linear(hf_layer.self_attn_layer_norm, pt_layer[0][0])
    _copy_linear(hf_layer.final_layer_norm, pt_layer[1][0])
    _copy_attn_layer(hf_layer.self_attn, pt_layer[0][1])
    pt_mlp = pt_layer[1][1]
    _copy_linear(hf_layer.fc1, pt_mlp.net[0][0])
    _copy_linear(hf_layer.fc2, pt_mlp.net[2])
