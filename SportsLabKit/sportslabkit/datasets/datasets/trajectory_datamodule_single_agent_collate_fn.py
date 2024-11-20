def single_agent_collate_fn(batch):
    x = torch.Tensor([seq for item in batch for seq in rearrange(item[0],
        'L N D ->  N L D')])
    y = torch.Tensor([seq for item in batch for seq in rearrange(item[1],
        'L N D ->  N L D')])
    return x, y
