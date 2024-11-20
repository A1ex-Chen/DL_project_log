def multi_agent_collate_fn(batch, max_num_agents, dummy_value=-1000):
    x_len = batch[0][0].shape[0]
    y_len = batch[0][1].shape[0]
    x = np.full((len(batch), x_len, max_num_agents, 2), dummy_value)
    y = np.full((len(batch), y_len, max_num_agents, 2), dummy_value)
    for i, (x_seq, y_seq) in enumerate(batch):
        num_agents = x_seq.shape[1]
        x[i, :, :num_agents, :] = x_seq[:, :max_num_agents, :]
        y[i, :, :num_agents, :] = y_seq[:, :max_num_agents, :]
    return torch.Tensor(x), torch.Tensor(y)
