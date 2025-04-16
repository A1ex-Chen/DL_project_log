def compute_implicit_phi(explicit_phi, f_n, k):
    k = min(len(explicit_phi) + 1, k)
    implicit_phi = collections.deque(maxlen=k)
    implicit_phi.append(f_n)
    for j in range(1, k):
        implicit_phi.append(tuple(iphi_ - ephi_ for iphi_, ephi_ in zip(
            implicit_phi[j - 1], explicit_phi[j - 1])))
    return implicit_phi
