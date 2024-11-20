def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.
        batch_time), args.batch_size, replace=False))
    batch_y0 = true_y[s]
    batch_t = t[:args.batch_time]
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)],
        dim=0)
    return batch_y0, batch_t, batch_y
