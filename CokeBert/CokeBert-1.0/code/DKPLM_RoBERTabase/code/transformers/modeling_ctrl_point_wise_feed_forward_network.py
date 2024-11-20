def point_wise_feed_forward_network(d_model_size, dff):
    return torch.nn.Sequential(torch.nn.Linear(d_model_size, dff), torch.nn
        .ReLU(), torch.nn.Linear(dff, d_model_size))
