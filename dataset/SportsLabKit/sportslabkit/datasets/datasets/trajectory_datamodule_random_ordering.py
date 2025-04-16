def random_ordering(data):
    num_agents = data.shape[1]
    data = data[:, torch.randperm(num_agents), :]
    return data
