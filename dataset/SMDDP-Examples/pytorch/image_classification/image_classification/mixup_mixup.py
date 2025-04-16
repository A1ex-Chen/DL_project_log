def mixup(alpha, data, target):
    with torch.no_grad():
        bs = data.size(0)
        c = np.random.beta(alpha, alpha)
        perm = torch.randperm(bs).cuda()
        md = c * data + (1 - c) * data[perm, :]
        mt = c * target + (1 - c) * target[perm, :]
        return md, mt
