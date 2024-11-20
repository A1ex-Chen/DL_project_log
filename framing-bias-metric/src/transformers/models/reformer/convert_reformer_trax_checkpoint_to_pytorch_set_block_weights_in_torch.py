def set_block_weights_in_torch(weights, torch_block, hidden_size):
    layer_norm_1 = weights[0][0][0]
    layer_norm_1_weight = np.asarray(layer_norm_1[0])
    layer_norm_1_bias = np.asarray(layer_norm_1[1])
    set_param(torch_block.attention.layer_norm, torch.tensor(
        layer_norm_1_weight), torch.tensor(layer_norm_1_bias))
    attn_weights = weights[0][1]
    if len(attn_weights) < 4:
        set_layer_weights_in_torch_lsh(attn_weights, torch_block.attention,
            hidden_size)
    else:
        set_layer_weights_in_torch_local(attn_weights, torch_block.
            attention, hidden_size)
    intermediate_weights = weights[2][0][1][2]
    if len(intermediate_weights) == 4:
        intermediate_weights = intermediate_weights[2]
    layer_norm_2_weight = np.asarray(intermediate_weights[0][0])
    layer_norm_2_bias = np.asarray(intermediate_weights[0][1])
    set_param(torch_block.feed_forward.layer_norm, torch.tensor(
        layer_norm_2_weight), torch.tensor(layer_norm_2_bias))
    inter_dense_weight = np.asarray(intermediate_weights[1][0])
    inter_dense_bias = np.asarray(intermediate_weights[1][1])
    set_param(torch_block.feed_forward.dense.dense, torch.tensor(
        inter_dense_weight).transpose(0, 1).contiguous(), torch.tensor(
        inter_dense_bias))
    out_dense_weight = np.asarray(intermediate_weights[4][0])
    out_dense_bias = np.asarray(intermediate_weights[4][1])
    set_param(torch_block.feed_forward.output.dense, torch.tensor(
        out_dense_weight).transpose(0, 1).contiguous(), torch.tensor(
        out_dense_bias))
