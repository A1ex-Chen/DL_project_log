def svdForRecom(datamat, user, simfunc, item):
    n = datamat.shape[1]
    similarity = 0
    prat = 0
    U, sigma, VT = np.linalg.svd(datamat)
    k = calcDim(sigma)
    k_simga = np.mat(np.diag(sigma[:k + 1]))
    k_datamat = datamat.T * U[:, :k + 1] * k_simga.I
    for j in range(n):
        a = datamat[user, j]
        if a == 0 or j == item:
            continue
        sim = simfunc(k_datamat[item, :].T, k_datamat[j, :].T)
        similarity += sim
        prat += sim * a
    if similarity == 0:
        return 0
    else:
        return prat / similarity
