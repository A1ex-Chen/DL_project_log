def sample_from_discretized_mix_logistic_1d(m, nr_mix):
    m = m.permute(0, 2, 3, 1)
    ls = [int(y) for y in m.size()]
    xs = ls[:-1] + [1]
    logit_probs = m[:, :, :, :nr_mix]
    m = m[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])
    temp = torch.FloatTensor(logit_probs.size())
    if m.is_cuda:
        temp = temp.cuda()
    temp.uniform_(1e-05, 1.0 - 1e-05)
    temp = logit_probs.data - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=3)
    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    means = torch.sum(m[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(m[:, :, :, :, nr_mix:2 * nr_mix] *
        sel, dim=4), min=-7.0)
    u = torch.FloatTensor(means.size())
    if m.is_cuda:
        u = u.cuda()
    u.uniform_(1e-05, 1.0 - 1e-05)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.0), max=1.0)
    out = x0.unsqueeze(1)
    return out
