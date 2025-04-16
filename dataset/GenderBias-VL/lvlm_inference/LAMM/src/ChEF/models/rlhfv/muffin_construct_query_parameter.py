def construct_query_parameter(query_k, h_size, init_weights, init=True):
    query_data = torch.zeros(query_k, h_size)
    if init:
        trunc_normal_(query_data, std=0.02)
    for idx in range(query_k):
        if init_weights[idx] is not None:
            query_data[idx] = init_weights[idx]
    query = torch.nn.Parameter(query_data)
    return query
