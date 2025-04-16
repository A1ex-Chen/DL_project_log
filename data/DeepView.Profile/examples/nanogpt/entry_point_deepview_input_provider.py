def deepview_input_provider(batch_size=48):
    data = np.random.randint(vocab_size, size=(batch_size, block_size + 1))
    x = torch.stack([torch.from_numpy(data[i, :-1].astype(np.int64)) for i in
        range(batch_size)])
    y = torch.stack([torch.from_numpy(data[i, 1:].astype(np.int64)) for i in
        range(batch_size)])
    return x.to(device), y.to(device)
