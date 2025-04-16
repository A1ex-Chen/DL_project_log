def g_and_explicit_phi(prev_t, next_t, implicit_phi, k):
    curr_t = prev_t[0]
    dt = next_t - prev_t[0]
    g = torch.empty(k + 1).to(prev_t[0])
    explicit_phi = collections.deque(maxlen=k)
    beta = torch.tensor(1).to(prev_t[0])
    g[0] = 1
    c = 1 / torch.arange(1, k + 2).to(prev_t[0])
    explicit_phi.append(implicit_phi[0])
    for j in range(1, k):
        beta = (next_t - prev_t[j - 1]) / (curr_t - prev_t[j]) * beta
        beat_cast = beta.to(implicit_phi[j][0])
        explicit_phi.append(tuple(iphi_ * beat_cast for iphi_ in
            implicit_phi[j]))
        c = c[:-1] - c[1:] if j == 1 else c[:-1] - c[1:] * dt / (next_t -
            prev_t[j - 1])
        g[j] = c[0]
    c = c[:-1] - c[1:] * dt / (next_t - prev_t[k - 1])
    g[k] = c[0]
    return g, explicit_phi
