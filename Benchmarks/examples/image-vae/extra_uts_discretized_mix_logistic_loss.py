def discretized_mix_logistic_loss(x, lh):
    """log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval"""
    x = x.permute(0, 2, 3, 1)
    lh = lh.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in lh.size()]
    nr_mix = int(ls[-1] / 10)
    logit_probs = lh[:, :, :, :nr_mix]
    lh = lh[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    means = lh[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(lh[:, :, :, :, nr_mix:2 * nr_mix], min=-7.0)
    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(),
        requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]
        ).view(xs[0], xs[1], xs[2], 1, nr_mix)
    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
        coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2],
        1, nr_mix)
    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = F.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
    inner_inner_cond = (cdf_delta > 1e-05).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta,
        min=1e-12)) + (1.0 - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond
        ) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
    return -torch.sum(log_sum_exp(log_probs))
