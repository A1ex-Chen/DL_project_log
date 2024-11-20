def set_layer_weights_in_torch_lsh(weights, torch_layer, hidden_size):
    np_query_key = np.asarray(weights[0])
    np_value = np.asarray(weights[1])
    np_dense = np.asarray(weights[2])
    set_param(torch_layer.self_attention.query_key, torch.tensor(
        np_query_key).transpose(1, 2).contiguous().view(-1, hidden_size))
    set_param(torch_layer.self_attention.value, torch.tensor(np_value).
        transpose(1, 2).contiguous().view(-1, hidden_size))
    set_param(torch_layer.output.dense, torch.tensor(np_dense).view(-1,
        hidden_size).contiguous().transpose(0, 1))
