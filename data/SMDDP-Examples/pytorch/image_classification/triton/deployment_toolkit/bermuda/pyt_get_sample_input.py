def get_sample_input(dataloader, device):
    for batch in dataloader:
        _, x, _ = batch
        break
    if isinstance(x, dict):
        sample_input = list(x.values())
    elif isinstance(x, list):
        sample_input = x
    else:
        raise TypeError(
            'The first element (x) of batch returned by dataloader must be a list or a dict'
            )
    for idx, s in enumerate(sample_input):
        sample_input[idx] = torch.from_numpy(s).to(device)
    return tuple(sample_input)
