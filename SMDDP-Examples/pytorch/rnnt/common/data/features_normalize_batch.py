@torch.jit.script
def normalize_batch(x, x_lens, normalize_type: str):
    if normalize_type == 'per_feature':
        mean = x.new_zeros(x.size(0), x.size(1))
        std = x.new_zeros(x.size(0), x.size(1))
        for i in range(x.size(0)):
            mean[i, :] = x[i, :, :x_lens[i]].mean(dim=1)
            std[i, :] = x[i, :, :x_lens[i]].std(dim=1)
        return (x - mean.unsqueeze(2)) / (std.unsqueeze(2) + 1e-05)
    elif normalize_type == 'all_features':
        mean = x.new_zeros(x.size(0))
        std = x.new_zeros(x.size(0))
        for i in range(x.size(0)):
            mean[i] = x[i, :, :x_lens[i]].mean()
            std[i] = x[i, :, :x_lens[i]].std()
        return (x - mean.view(-1, 1, 1)) / (std.view(-1, 1, 1) + 1e-05)
    else:
        return x
