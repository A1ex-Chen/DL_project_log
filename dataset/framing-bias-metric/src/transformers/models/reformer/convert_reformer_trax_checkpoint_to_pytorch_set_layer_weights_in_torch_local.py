def set_layer_weights_in_torch_local(weights, torch_layer, hidden_size):
    np_query = np.asarray(weights[0])
    np_key = np.asarray(weights[1])
    np_value = np.asarray(weights[2])
    np_dense = np.asarray(weights[3])
    set_param(torch_layer.self_attention.query, torch.tensor(np_query).
        transpose(1, 2).contiguous().view(-1, hidden_size))
    set_param(torch_layer.self_attention.key, torch.tensor(np_key).
        transpose(1, 2).contiguous().view(-1, hidden_size))
    set_param(torch_layer.self_attention.value, torch.tensor(np_value).
        transpose(1, 2).contiguous().view(-1, hidden_size))
    set_param(torch_layer.output.dense, torch.tensor(np_dense).view(-1,
        hidden_size).contiguous().transpose(0, 1))
