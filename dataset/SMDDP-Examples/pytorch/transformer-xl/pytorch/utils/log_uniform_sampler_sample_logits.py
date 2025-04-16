def sample_logits(embedding, bias, labels, inputs, sampler):
    """
        embedding: an nn.Embedding layer
        bias: [n_vocab]
        labels: [b1, b2]
        inputs: [b1, b2, n_emb]
        sampler: you may use a LogUniformSampler
    Return
        logits: [b1, b2, 1 + n_sample]
    """
    true_log_probs, samp_log_probs, neg_samples = sampler.sample(labels)
    n_sample = neg_samples.size(0)
    b1, b2 = labels.size(0), labels.size(1)
    all_ids = torch.cat([labels.view(-1), neg_samples])
    all_w = embedding(all_ids)
    true_w = all_w[:-n_sample].view(b1, b2, -1)
    sample_w = all_w[-n_sample:].view(n_sample, -1)
    all_b = bias[all_ids]
    true_b = all_b[:-n_sample].view(b1, b2)
    sample_b = all_b[-n_sample:]
    hit = (labels[:, :, None] == neg_samples).detach()
    true_logits = torch.einsum('ijk,ijk->ij', [true_w, inputs]
        ) + true_b - true_log_probs
    sample_logits = torch.einsum('lk,ijk->ijl', [sample_w, inputs]
        ) + sample_b - samp_log_probs
    sample_logits.masked_fill_(hit, -1e+30)
    logits = torch.cat([true_logits[:, :, None], sample_logits], -1)
    return logits
